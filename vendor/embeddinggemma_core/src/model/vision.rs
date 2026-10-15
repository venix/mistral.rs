//! The Gemma 4 vision encoder (`Gemma4VisionModel`) and `embed_vision`:
//!
//! - patches of 16×16×3 pixels in `[0, 1]`, mapped to `2 (x - 0.5)` and
//!   projected 768 → 768, plus learned x / y position embeddings;
//! - 16 bidirectional layers (12 heads × 64): q/k RMSNorms, a scale-free v
//!   RMSNorm, axial 2-D RoPE (θ = 100, first half of each head rotated by
//!   the x position, second half by y), attention scale 1.0, sandwich norms,
//!   GELU-tanh MLP;
//! - 3×3 average pooling over the patch grid (one soft token per 48×48
//!   pixels), × √hidden in f32;
//! - `embed_vision`: scale-free RMSNorm, then 768 → text hidden.

use super::{ClippedLinear, Linear, Loader};
use crate::backend::{AttentionMask, Backend};
use crate::config::VisionConfig;
use crate::device::Precision;
use crate::error::{EmbedError, Result};
use crate::media::ImagePatches;

struct Layer<T> {
    input_layernorm: T,
    q_proj: ClippedLinear<T>,
    k_proj: ClippedLinear<T>,
    v_proj: ClippedLinear<T>,
    o_proj: ClippedLinear<T>,
    q_norm: T,
    k_norm: T,
    post_attention_layernorm: T,
    pre_feedforward_layernorm: T,
    gate_proj: ClippedLinear<T>,
    up_proj: ClippedLinear<T>,
    down_proj: ClippedLinear<T>,
    post_feedforward_layernorm: T,
}

pub struct VisionTower<B: Backend> {
    input_proj: Linear<B::Tensor>,
    /// `position_embedding_table[0]` / `[1]`: `[positions, hidden]`.
    pos_x: B::Tensor,
    pos_y: B::Tensor,
    layers: Vec<Layer<B::Tensor>>,
    embed_projection: Linear<B::Tensor>,
    cfg: VisionConfig,
}

const PREFIX: &str = "vision_tower";

impl<B: Backend> VisionTower<B> {
    pub fn load(w: &Loader<'_, B>, cfg: &VisionConfig, text_hidden: usize) -> Result<Self> {
        if cfg.standardize {
            return Err(EmbedError::Config(
                "vision standardize = true is not supported".into(),
            ));
        }
        if let Some(t) = &cfg.rope_parameters.rope_type
            && t != "axial"
        {
            return Err(EmbedError::Config(format!(
                "unsupported vision rope_type `{t}` (only axial)"
            )));
        }
        if cfg.num_key_value_heads != cfg.num_attention_heads {
            return Err(EmbedError::Config("vision GQA is not supported".into()));
        }
        let be = w.be;
        let h = cfg.hidden_size;
        let p = |name: &str| format!("{PREFIX}.{name}");
        let patch_dim = 3 * cfg.patch_size * cfg.patch_size;
        let table = w.get(
            &p("patch_embedder.position_embedding_table"),
            &[2, cfg.position_embedding_size, h],
        )?;
        let pos = |i: usize| -> Result<B::Tensor> {
            be.reshape(
                &be.narrow(&table, 0, i, 1)?,
                &[cfg.position_embedding_size, h],
            )
        };
        let qkv = cfg.num_attention_heads * cfg.head_dim;
        let clip = cfg.use_clipped_linears;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let l = |name: &str| p(&format!("encoder.layers.{i}.{name}"));
                Ok(Layer {
                    input_layernorm: w.get_f32(&l("input_layernorm.weight"), &[h])?,
                    q_proj: w.clipped(&l("self_attn.q_proj"), qkv, h, clip)?,
                    k_proj: w.clipped(&l("self_attn.k_proj"), qkv, h, clip)?,
                    v_proj: w.clipped(&l("self_attn.v_proj"), qkv, h, clip)?,
                    o_proj: w.clipped(&l("self_attn.o_proj"), h, qkv, clip)?,
                    q_norm: w.get_f32(&l("self_attn.q_norm.weight"), &[cfg.head_dim])?,
                    k_norm: w.get_f32(&l("self_attn.k_norm.weight"), &[cfg.head_dim])?,
                    post_attention_layernorm: w
                        .get_f32(&l("post_attention_layernorm.weight"), &[h])?,
                    pre_feedforward_layernorm: w
                        .get_f32(&l("pre_feedforward_layernorm.weight"), &[h])?,
                    gate_proj: w.clipped(&l("mlp.gate_proj"), cfg.intermediate_size, h, clip)?,
                    up_proj: w.clipped(&l("mlp.up_proj"), cfg.intermediate_size, h, clip)?,
                    down_proj: w.clipped(&l("mlp.down_proj"), h, cfg.intermediate_size, clip)?,
                    post_feedforward_layernorm: w
                        .get_f32(&l("post_feedforward_layernorm.weight"), &[h])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input_proj: w.linear(&p("patch_embedder.input_proj"), h, patch_dim)?,
            pos_x: pos(0)?,
            pos_y: pos(1)?,
            layers,
            embed_projection: w.linear("embed_vision.embedding_projection", text_hidden, h)?,
            cfg: cfg.clone(),
        })
    }

    /// One image (or video frame) → soft tokens `[ph/k × pw/k, text_hidden]`.
    pub fn forward(&self, be: &B, img: &ImagePatches) -> Result<B::Tensor> {
        let cfg = &self.cfg;
        let k = cfg.pooling_kernel_size;
        let (ph, pw) = (img.patches_h, img.patches_w);
        let n = ph * pw;
        let patch_dim = 3 * cfg.patch_size * cfg.patch_size;
        if img.pixels.len() != n * patch_dim || ph % k != 0 || pw % k != 0 {
            return Err(EmbedError::InvalidRequest(format!(
                "image patches {ph}×{pw} do not fit the {k}×{k} pooling grid"
            )));
        }
        if ph.max(pw) > cfg.position_embedding_size {
            return Err(EmbedError::InvalidRequest(format!(
                "image is {ph}×{pw} patches; the position table has {}",
                cfg.position_embedding_size
            )));
        }
        let hidden = cfg.hidden_size;
        let precision = be.precision();

        // `Gemma4VisionPatchEmbedder`: 2 (x - 0.5), project, add positions.
        let pixels: Vec<f32> = img.pixels.iter().map(|x| 2.0 * (x - 0.5)).collect();
        let pixels = be.cast(&be.f32_tensor(&pixels, &[1, n, patch_dim])?, precision)?;
        let xs: Vec<u32> = (0..n).map(|i| (i % pw) as u32).collect();
        let ys: Vec<u32> = (0..n).map(|i| (i / pw) as u32).collect();
        let xs_t = be.u32_tensor(&xs, &[1, n])?;
        let ys_t = be.u32_tensor(&ys, &[1, n])?;
        let pos = be.add(
            &be.embedding(&self.pos_x, &xs_t)?,
            &be.embedding(&self.pos_y, &ys_t)?,
        )?;
        let mut x = be.add(&self.input_proj.forward(be, &pixels)?, &pos)?;

        // Axial RoPE: head_dim / 2 channels per axis.
        let half = cfg.head_dim / 2;
        let (cos_x, sin_x) = self.rope_table(be, &xs, half)?;
        let (cos_y, sin_y) = self.rope_table(be, &ys, half)?;
        let mask = AttentionMask {
            lengths: vec![n],
            seq_len: n,
            window: None,
        };
        let eps = cfg.rms_norm_eps;
        let heads = cfg.num_attention_heads;
        let hd = cfg.head_dim;
        let rope2d = |t: &B::Tensor| -> Result<B::Tensor> {
            let a = be.rope(&be.narrow(t, 3, 0, half)?, &cos_x, &sin_x)?;
            let b = be.rope(&be.narrow(t, 3, half, half)?, &cos_y, &sin_y)?;
            be.cat(&[a, b], 3)
        };
        for l in &self.layers {
            let h = be.rms_norm(&x, Some(&l.input_layernorm), eps)?;
            let split = |t: B::Tensor| be.reshape(&t, &[1, n, heads, hd]);
            let q = split(l.q_proj.forward(be, &h)?)?;
            let k_ = split(l.k_proj.forward(be, &h)?)?;
            let v = split(l.v_proj.forward(be, &h)?)?;
            let q = rope2d(&be.transpose(&be.rms_norm(&q, Some(&l.q_norm), eps)?, 1, 2)?)?;
            let k_ = rope2d(&be.transpose(&be.rms_norm(&k_, Some(&l.k_norm), eps)?, 1, 2)?)?;
            let v = be.transpose(&be.rms_norm(&v, None, eps)?, 1, 2)?;
            let attn = be.attention(&q, &k_, &v, &mask, 1.0)?;
            let h = l.o_proj.forward(be, &attn)?;
            let h = be.rms_norm(&h, Some(&l.post_attention_layernorm), eps)?;
            x = be.add(&x, &h)?;

            let h = be.rms_norm(&x, Some(&l.pre_feedforward_layernorm), eps)?;
            let gate = be.gelu_tanh(&l.gate_proj.forward(be, &h)?)?;
            let up = l.up_proj.forward(be, &h)?;
            let h = l.down_proj.forward(be, &be.mul(&gate, &up)?)?;
            let h = be.rms_norm(&h, Some(&l.post_feedforward_layernorm), eps)?;
            x = be.add(&x, &h)?;
        }

        // `Gemma4VisionPooler`: k×k average in f32 (patches are row-major,
        // index = y·pw + x), back to the working dtype, then ×√hidden in f32.
        let (gh, gw) = (ph / k, pw / k);
        let x = be.cast(&x, Precision::F32)?;
        let x = be.reshape(&x, &[gh, k, gw, k, hidden])?;
        let x = be.sum(&be.sum(&x, 3, false)?, 1, false)?; // [gh, gw, hidden]
        let x = be.affine(&x, 1.0 / (k * k) as f64, 0.0)?;
        let x = be.cast(&be.cast(&x, precision)?, Precision::F32)?;
        let x = be.affine(&x, (hidden as f64).sqrt(), 0.0)?;
        let x = be.cast(&be.reshape(&x, &[gh * gw, hidden])?, precision)?;

        // `embed_vision`.
        let x = be.rms_norm(&x, None, eps)?;
        self.embed_projection.forward(be, &x)
    }

    /// `cos` / `sin` `[n, half/2]` for one axis: `inv_freq_i = θ^(-2i/half)`.
    fn rope_table(&self, be: &B, pos: &[u32], half: usize) -> Result<(B::Tensor, B::Tensor)> {
        let theta = self.cfg.rope_parameters.rope_theta;
        let quarter = half / 2;
        let inv: Vec<f32> = (0..quarter)
            .map(|i| (1.0 / theta.powf((2 * i) as f64 / half as f64)) as f32)
            .collect();
        let mut cos = Vec::with_capacity(pos.len() * quarter);
        let mut sin = Vec::with_capacity(pos.len() * quarter);
        for &p in pos {
            for &f in &inv {
                let a = p as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        let n = pos.len();
        let p = be.precision();
        Ok((
            be.cast(&be.f32_tensor(&cos, &[n, quarter])?, p)?,
            be.cast(&be.f32_tensor(&sin, &[n, quarter])?, p)?,
        ))
    }
}
