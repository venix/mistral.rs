//! The EmbeddingGemma 2 text encoder (`EmbeddingGemma2TextModel`):
//!
//! - token embeddings × √hidden (rounded to the weight dtype);
//! - projection-only per-layer embeddings (PLE) derived from the input
//!   embeddings — media soft tokens included;
//! - 24 bidirectional layers, 5 sliding-window (head_dim 256, 2 KV heads,
//!   θ = 10⁴, `|q - k| <= 512`) to 1 full (head_dim 512, 1 KV head, θ = 10⁶),
//!   each with q/k RMSNorms, a scale-free v RMSNorm, attention scale 1.0,
//!   sandwich norms around attention and the GELU-tanh MLP, a PLE gate and a
//!   learned `layer_scalar`;
//! - final RMSNorm and the 512 → 768 `embedding_projection` per token.
//!
//! RMSNorm is Gemma 4's `x * w` (not Gemma 3's `x * (1 + w)`).

use super::{Linear, Loader, round_to};
use crate::backend::{AttentionMask, Backend};
use crate::config::{TextConfig, TextLayerSpec};
use crate::error::{EmbedError, Result};

/// `cos` / `sin` tables `[max_positions, head_dim / 2]` in the compute
/// precision, frequencies in f32 like `transformers`.
pub struct RopeTable<T> {
    pub cos: T,
    pub sin: T,
    pub head_dim: usize,
    pub theta: f64,
}

impl<T: Clone> RopeTable<T> {
    pub fn new<B: Backend<Tensor = T>>(
        be: &B,
        head_dim: usize,
        theta: f64,
        max_positions: usize,
    ) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (1.0 / theta.powf((2 * i) as f64 / head_dim as f64)) as f32)
            .collect();
        let mut cos = Vec::with_capacity(max_positions * half);
        let mut sin = Vec::with_capacity(max_positions * half);
        for pos in 0..max_positions {
            for &f in &inv_freq {
                let angle = pos as f32 * f;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let p = be.precision();
        Ok(Self {
            cos: be.cast(&be.f32_tensor(&cos, &[max_positions, half])?, p)?,
            sin: be.cast(&be.f32_tensor(&sin, &[max_positions, half])?, p)?,
            head_dim,
            theta,
        })
    }

    /// Rotate `x [b, h, s, d]` for positions `0..s`.
    pub fn apply<B: Backend<Tensor = T>>(&self, be: &B, x: &T) -> Result<T> {
        let s = be.shape(x)[2];
        let max = be.shape(&self.cos)[0];
        if s > max {
            return Err(EmbedError::InvalidRequest(format!(
                "sequence of {s} tokens exceeds the {max}-position RoPE table"
            )));
        }
        let cos = be.narrow(&self.cos, 0, 0, s)?;
        let sin = be.narrow(&self.sin, 0, 0, s)?;
        be.rope(x, &cos, &sin)
    }
}

struct Attention<T> {
    q_proj: Linear<T>,
    k_proj: Linear<T>,
    v_proj: Linear<T>,
    o_proj: Linear<T>,
    q_norm: T,
    k_norm: T,
    spec: TextLayerSpec,
    heads: usize,
    /// Index into [`TextModel::ropes`].
    rope: usize,
}

struct Layer<T> {
    input_layernorm: T,
    attn: Attention<T>,
    post_attention_layernorm: T,
    pre_feedforward_layernorm: T,
    gate_proj: Linear<T>,
    up_proj: Linear<T>,
    down_proj: Linear<T>,
    post_feedforward_layernorm: T,
    ple_gate: Linear<T>,
    ple_projection: Linear<T>,
    ple_norm: T,
    layer_scalar: T,
}

pub struct TextModel<B: Backend> {
    embed_tokens: B::Tensor,
    embed_scale: f64,
    /// `per_layer_model_projection` split into one `[ple, hidden]` slice per
    /// layer, so the `[b, s, layers × ple]` tensor is never materialised.
    ple_projections: Vec<Linear<B::Tensor>>,
    ple_scale: f64,
    ple_norm: B::Tensor,
    layers: Vec<Layer<B::Tensor>>,
    norm: B::Tensor,
    embedding_projection: Linear<B::Tensor>,
    ropes: Vec<RopeTable<B::Tensor>>,
    sliding_window: usize,
    eps: f64,
    embedding_dim: usize,
}

const PREFIX: &str = "language_model";

impl<B: Backend> TextModel<B> {
    pub fn load(w: &Loader<'_, B>, cfg: &TextConfig, max_positions: usize) -> Result<Self> {
        if let Some(act) = &cfg.hidden_activation
            && act != "gelu_pytorch_tanh"
        {
            return Err(EmbedError::Config(format!(
                "unsupported text hidden_activation `{act}`"
            )));
        }
        let be = w.be;
        let h = cfg.hidden_size;
        let ple = cfg.hidden_size_per_layer_input;
        let n = cfg.num_hidden_layers;
        let specs = cfg.layer_specs()?;
        let p = |name: &str| format!("{PREFIX}.{name}");

        let mut ropes: Vec<RopeTable<B::Tensor>> = Vec::new();
        let mut rope_index = |spec: &TextLayerSpec| -> Result<usize> {
            if let Some(i) = ropes
                .iter()
                .position(|r| r.head_dim == spec.head_dim && r.theta == spec.rope_theta)
            {
                return Ok(i);
            }
            ropes.push(RopeTable::new(
                be,
                spec.head_dim,
                spec.rope_theta,
                max_positions,
            )?);
            Ok(ropes.len() - 1)
        };

        let projection = w.get(&p("ple.per_layer_model_projection.weight"), &[n * ple, h])?;
        let ple_projections = (0..n)
            .map(|i| {
                Ok(Linear {
                    w: be.narrow(&projection, 0, i * ple, ple)?,
                    b: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut layers = Vec::with_capacity(n);
        for (i, spec) in specs.iter().enumerate() {
            let l = |name: &str| p(&format!("layers.{i}.{name}"));
            let heads = cfg.num_attention_heads;
            let (hd, kv) = (spec.head_dim, spec.kv_heads);
            layers.push(Layer {
                input_layernorm: w.get_f32(&l("input_layernorm.weight"), &[h])?,
                attn: Attention {
                    q_proj: w.linear(&l("self_attn.q_proj"), heads * hd, h)?,
                    k_proj: w.linear(&l("self_attn.k_proj"), kv * hd, h)?,
                    v_proj: w.linear(&l("self_attn.v_proj"), kv * hd, h)?,
                    o_proj: w.linear(&l("self_attn.o_proj"), h, heads * hd)?,
                    q_norm: w.get_f32(&l("self_attn.q_norm.weight"), &[hd])?,
                    k_norm: w.get_f32(&l("self_attn.k_norm.weight"), &[hd])?,
                    spec: *spec,
                    heads,
                    rope: rope_index(spec)?,
                },
                post_attention_layernorm: w.get_f32(&l("post_attention_layernorm.weight"), &[h])?,
                pre_feedforward_layernorm: w
                    .get_f32(&l("pre_feedforward_layernorm.weight"), &[h])?,
                gate_proj: w.linear(&l("mlp.gate_proj"), cfg.intermediate_size, h)?,
                up_proj: w.linear(&l("mlp.up_proj"), cfg.intermediate_size, h)?,
                down_proj: w.linear(&l("mlp.down_proj"), h, cfg.intermediate_size)?,
                post_feedforward_layernorm: w
                    .get_f32(&l("post_feedforward_layernorm.weight"), &[h])?,
                ple_gate: w.linear(&l("ple_block.per_layer_input_gate"), ple, h)?,
                ple_projection: w.linear(&l("ple_block.per_layer_projection"), h, ple)?,
                ple_norm: w.get_f32(&l("ple_block.post_per_layer_input_norm.weight"), &[h])?,
                layer_scalar: w.get(&l("layer_scalar"), &[1])?,
            });
        }

        Ok(Self {
            embed_tokens: w.get(&p("embed_tokens.weight"), &[cfg.vocab_size, h])?,
            // `EmbeddingGemma2TextScaledWordEmbedding`: √hidden cast to the
            // weight dtype (22.625 in bf16) before the multiply.
            embed_scale: round_to(be, (h as f32).sqrt(), be.precision())?,
            ple_projections,
            ple_scale: (h as f64).powf(-0.5),
            ple_norm: w.get_f32(&p("ple.per_layer_projection_norm.weight"), &[ple])?,
            layers,
            norm: w.get_f32(&p("norm.weight"), &[h])?,
            embedding_projection: w.linear(&p("embedding_projection"), cfg.embedding_dim, h)?,
            ropes,
            sliding_window: cfg.sliding_window,
            eps: cfg.rms_norm_eps,
            embedding_dim: cfg.embedding_dim,
        })
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Scaled token embeddings for `ids [b, s]`.
    pub fn embed_tokens(&self, be: &B, ids: &B::Tensor) -> Result<B::Tensor> {
        be.affine(
            &be.embedding(&self.embed_tokens, ids)?,
            self.embed_scale,
            0.0,
        )
    }

    /// `inputs_embeds [b, s, hidden]` (right-padded rows of `lengths`) →
    /// per-token embeddings `[b, s, embedding_dim]`.
    pub fn forward(
        &self,
        be: &B,
        inputs_embeds: &B::Tensor,
        lengths: &[usize],
    ) -> Result<B::Tensor> {
        let seq_len = be.shape(inputs_embeds)[1];
        let full = AttentionMask {
            lengths: lengths.to_vec(),
            seq_len,
            window: None,
        };
        let sliding = AttentionMask {
            window: Some(self.sliding_window),
            ..full.clone()
        };
        let mut x = inputs_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            // Per-layer input, from the merged input embeddings.
            let ple = self.ple_projections[i].forward(be, inputs_embeds)?;
            let ple = be.rms_norm(
                &be.affine(&ple, self.ple_scale, 0.0)?,
                Some(&self.ple_norm),
                self.eps,
            )?;
            let mask = if layer.attn.spec.sliding {
                &sliding
            } else {
                &full
            };
            x = self.layer(be, layer, &x, &ple, mask)?;
        }
        let x = be.rms_norm(&x, Some(&self.norm), self.eps)?;
        self.embedding_projection.forward(be, &x)
    }

    fn layer(
        &self,
        be: &B,
        l: &Layer<B::Tensor>,
        x: &B::Tensor,
        ple: &B::Tensor,
        mask: &AttentionMask,
    ) -> Result<B::Tensor> {
        let h = be.rms_norm(x, Some(&l.input_layernorm), self.eps)?;
        let h = self.attention(be, &l.attn, &h, mask)?;
        let h = be.rms_norm(&h, Some(&l.post_attention_layernorm), self.eps)?;
        let x = be.add(x, &h)?;

        let h = be.rms_norm(&x, Some(&l.pre_feedforward_layernorm), self.eps)?;
        let gate = be.gelu_tanh(&l.gate_proj.forward(be, &h)?)?;
        let up = l.up_proj.forward(be, &h)?;
        let h = l.down_proj.forward(be, &be.mul(&gate, &up)?)?;
        let h = be.rms_norm(&h, Some(&l.post_feedforward_layernorm), self.eps)?;
        let x = be.add(&x, &h)?;

        let g = be.gelu_tanh(&l.ple_gate.forward(be, &x)?)?;
        let g = l.ple_projection.forward(be, &be.mul(&g, ple)?)?;
        let g = be.rms_norm(&g, Some(&l.ple_norm), self.eps)?;
        let x = be.add(&x, &g)?;
        be.mul(&x, &l.layer_scalar)
    }

    fn attention(
        &self,
        be: &B,
        a: &Attention<B::Tensor>,
        x: &B::Tensor,
        mask: &AttentionMask,
    ) -> Result<B::Tensor> {
        let shape = be.shape(x);
        let (b, s) = (shape[0], shape[1]);
        let hd = a.spec.head_dim;
        let heads = |t: &B::Tensor, n: usize| be.reshape(t, &[b, s, n, hd]);
        let q = heads(&a.q_proj.forward(be, x)?, a.heads)?;
        let k = heads(&a.k_proj.forward(be, x)?, a.spec.kv_heads)?;
        let v = heads(&a.v_proj.forward(be, x)?, a.spec.kv_heads)?;
        let q = be.transpose(&be.rms_norm(&q, Some(&a.q_norm), self.eps)?, 1, 2)?;
        let k = be.transpose(&be.rms_norm(&k, Some(&a.k_norm), self.eps)?, 1, 2)?;
        let v = be.transpose(&be.rms_norm(&v, None, self.eps)?, 1, 2)?;
        let rope = &self.ropes[a.rope];
        let q = rope.apply(be, &q)?;
        let k = rope.apply(be, &k)?;
        let out = be.attention(&q, &k, &v, mask, 1.0)?;
        a.o_proj.forward(be, &out)
    }
}
