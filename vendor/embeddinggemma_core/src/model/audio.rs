//! The Gemma 4 audio encoder (`Gemma4AudioModel`, a USM-style Conformer)
//! and `embed_audio`:
//!
//! - log-mel features `[frames, 128]` → two 3×3 stride-2 convolutions
//!   (channels 128, 32; LayerNorm + ReLU each) → 1024-d tokens at 40 ms;
//! - 12 Conformer layers: ½·FFN → local self-attention → light depthwise
//!   conv (kernel 5, causal) → ½·FFN → RMSNorm. Every projection is a
//!   `ClippableLinear` with learned input/output bounds;
//! - the attention is chunked local attention: each token sees itself and
//!   the 11 tokens before it (`attention_context_left - 1` keys, no right
//!   context), logits = `q·k + q·r(t - s)` with a sinusoidal relative
//!   position table projected by `relative_k_proj`, soft-capped at 50;
//! - `output_proj` 1024 → 1536, then `embed_audio`: scale-free RMSNorm and
//!   1536 → text hidden.
//!
//! The reference computes the attention block-wise (12-token chunks with a
//! 24-token context); every logit outside the window is masked, so the dense
//! banded form here is the same function.

use super::{ClippedLinear, Linear, Loader};
use crate::backend::Backend;
use crate::config::AudioConfig;
use crate::device::Precision;
use crate::error::{EmbedError, Result};
use crate::media::AudioFeatures;

struct FeedForward<T> {
    pre_norm: T,
    ffw1: ClippedLinear<T>,
    ffw2: ClippedLinear<T>,
    post_norm: T,
}

struct Attention<T> {
    q_proj: ClippedLinear<T>,
    k_proj: ClippedLinear<T>,
    v_proj: ClippedLinear<T>,
    post: ClippedLinear<T>,
    relative_k_proj: Linear<T>,
    /// `q_scale · softplus(per_dim_scale)`, `[head_dim]`, f32.
    q_scale: T,
}

struct LightConv<T> {
    pre_norm: T,
    linear_start: ClippedLinear<T>,
    /// Depthwise kernel taps `[kernel][channels]`, compute precision.
    taps: Vec<T>,
    conv_norm: T,
    linear_end: ClippedLinear<T>,
}

struct Layer<T> {
    ff1: FeedForward<T>,
    norm_pre_attn: T,
    attn: Attention<T>,
    norm_post_attn: T,
    lconv: LightConv<T>,
    ff2: FeedForward<T>,
    norm_out: T,
}

pub struct AudioTower<B: Backend> {
    conv0: B::Tensor,
    norm0: B::Tensor,
    conv1: B::Tensor,
    norm1: B::Tensor,
    input_proj: Linear<B::Tensor>,
    layers: Vec<Layer<B::Tensor>>,
    output_proj: Linear<B::Tensor>,
    embed_projection: Linear<B::Tensor>,
    /// Sinusoidal relative positions `[context/2 + 1, hidden]` for
    /// distances `context/2 … 0`, compute precision.
    rel_pos: B::Tensor,
    cfg: AudioConfig,
}

const PREFIX: &str = "audio_tower";

impl<B: Backend> AudioTower<B> {
    pub fn load(w: &Loader<'_, B>, cfg: &AudioConfig, text_hidden: usize) -> Result<Self> {
        if let Some(act) = &cfg.hidden_act
            && act != "silu"
        {
            return Err(EmbedError::Config(format!(
                "unsupported audio hidden_act `{act}`"
            )));
        }
        if cfg.subsampling_conv_channels.len() != 2 || cfg.attention_context_right != 0 {
            return Err(EmbedError::Config(
                "unsupported audio subsampling / attention context layout".into(),
            ));
        }
        let be = w.be;
        let h = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let head_dim = h / heads;
        let (c0, c1) = (
            cfg.subsampling_conv_channels[0],
            cfg.subsampling_conv_channels[1],
        );
        let clip = cfg.use_clipped_linears;
        let p = |name: &str| format!("{PREFIX}.{name}");

        // q_scale = head_dim^-½ / ln 2, times softplus(per_dim_scale) taken in
        // the weight dtype like the reference.
        let q_base = (head_dim as f64).powf(-0.5) / std::f64::consts::LN_2;
        let precision = be.precision();

        let ff = |prefix: &str| -> Result<FeedForward<B::Tensor>> {
            Ok(FeedForward {
                pre_norm: w.get_f32(&format!("{prefix}.pre_layer_norm.weight"), &[h])?,
                ffw1: w.clipped(&format!("{prefix}.ffw_layer_1"), 4 * h, h, clip)?,
                ffw2: w.clipped(&format!("{prefix}.ffw_layer_2"), h, 4 * h, clip)?,
                post_norm: w.get_f32(&format!("{prefix}.post_layer_norm.weight"), &[h])?,
            })
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let l = |name: &str| p(&format!("layers.{i}.{name}"));
            let per_dim = w.get(&l("self_attn.per_dim_scale"), &[head_dim])?;
            let per_dim = be.to_f32_vec(&per_dim)?;
            let softplus: Vec<f32> = per_dim
                .iter()
                .map(|&x| (x as f64).exp().ln_1p() as f32)
                .collect();
            let softplus = be.cast(&be.f32_tensor(&softplus, &[head_dim])?, precision)?;
            let q_scale = be.affine(&be.cast(&softplus, Precision::F32)?, q_base, 0.0)?;

            let dw = w.get(
                &l("lconv1d.depthwise_conv1d.weight"),
                &[h, 1, cfg.conv_kernel_size],
            )?;
            let dw = be.reshape(&dw, &[h, cfg.conv_kernel_size])?;
            let taps = (0..cfg.conv_kernel_size)
                .map(|t| be.reshape(&be.narrow(&dw, 1, t, 1)?, &[h]))
                .collect::<Result<Vec<_>>>()?;

            layers.push(Layer {
                ff1: ff(&l("feed_forward1"))?,
                norm_pre_attn: w.get_f32(&l("norm_pre_attn.weight"), &[h])?,
                attn: Attention {
                    q_proj: w.clipped(&l("self_attn.q_proj"), h, h, clip)?,
                    k_proj: w.clipped(&l("self_attn.k_proj"), h, h, clip)?,
                    v_proj: w.clipped(&l("self_attn.v_proj"), h, h, clip)?,
                    post: w.clipped(&l("self_attn.post"), h, h, clip)?,
                    relative_k_proj: w.linear(&l("self_attn.relative_k_proj"), h, h)?,
                    q_scale,
                },
                norm_post_attn: w.get_f32(&l("norm_post_attn.weight"), &[h])?,
                lconv: LightConv {
                    pre_norm: w.get_f32(&l("lconv1d.pre_layer_norm.weight"), &[h])?,
                    linear_start: w.clipped(&l("lconv1d.linear_start"), 2 * h, h, clip)?,
                    taps,
                    conv_norm: w.get_f32(&l("lconv1d.conv_norm.weight"), &[h])?,
                    linear_end: w.clipped(&l("lconv1d.linear_end"), h, h, clip)?,
                },
                ff2: ff(&l("feed_forward2"))?,
                norm_out: w.get_f32(&l("norm_out.weight"), &[h])?,
            });
        }

        // Gemma4AudioRelPositionalEncoding: positions context/2 … 0,
        // [sin…, cos…] with log-spaced timescales 1 … 10⁴.
        let context =
            cfg.attention_chunk_size + cfg.attention_context_left - 1 + cfg.attention_context_right;
        let n_pos = context / 2 + 1;
        let half = h / 2;
        let incr = (10_000f64).ln() / ((half as f64) - 1.0).max(1.0);
        let inv: Vec<f32> = (0..half)
            .map(|i| (-(i as f64) * incr).exp() as f32)
            .collect();
        let mut rel = Vec::with_capacity(n_pos * h);
        for r in 0..n_pos {
            let pos = (n_pos - 1 - r) as f32;
            rel.extend(inv.iter().map(|f| (pos * f).sin()));
            rel.extend(inv.iter().map(|f| (pos * f).cos()));
        }
        let rel_pos = be.cast(&be.f32_tensor(&rel, &[n_pos, h])?, precision)?;

        Ok(Self {
            conv0: w.get(
                &p("subsample_conv_projection.layer0.conv.weight"),
                &[c0, 1, 3, 3],
            )?,
            norm0: w.get_f32(&p("subsample_conv_projection.layer0.norm.weight"), &[c0])?,
            conv1: w.get(
                &p("subsample_conv_projection.layer1.conv.weight"),
                &[c1, c0, 3, 3],
            )?,
            norm1: w.get_f32(&p("subsample_conv_projection.layer1.norm.weight"), &[c1])?,
            input_proj: w.linear(
                &p("subsample_conv_projection.input_proj_linear"),
                h,
                // `(subsampling_conv_channels[0] // 4) * channels[1]` (128 mel
                // bins / 4 after two stride-2 convs × 32 channels).
                (c0 / 4) * c1,
            )?,
            layers,
            output_proj: w.linear_bias(&p("output_proj"), cfg.output_proj_dims, h)?,
            embed_projection: w.linear(
                "embed_audio.embedding_projection",
                text_hidden,
                cfg.output_proj_dims,
            )?,
            rel_pos,
            cfg: cfg.clone(),
        })
    }

    /// One clip of log-mel features → soft tokens `[tokens, text_hidden]`.
    pub fn forward(&self, be: &B, a: &AudioFeatures) -> Result<B::Tensor> {
        let precision = be.precision();
        let bins = a.bins;
        if a.mel.len() != a.frames * bins || a.valid_frames > a.frames || a.valid_frames == 0 {
            return Err(EmbedError::InvalidRequest(
                "malformed audio features".into(),
            ));
        }
        let eps = self.cfg.rms_norm_eps;

        // Subsampling: [1, 1, T, F] → conv/LN/ReLU → mask → conv/LN/ReLU.
        let x = be.cast(&be.f32_tensor(&a.mel, &[1, 1, a.frames, bins])?, precision)?;
        let x = self.conv_block(be, &x, &self.conv0, &self.norm0)?;
        let t1 = be.shape(&x)[2];
        let valid1 = a.valid_frames.div_ceil(2);
        let time_mask: Vec<f32> = (0..t1)
            .map(|t| if t < valid1 { 1.0 } else { 0.0 })
            .collect();
        let time_mask = be.cast(&be.f32_tensor(&time_mask, &[1, 1, t1, 1])?, precision)?;
        let x = be.mul(&x, &time_mask)?;
        let x = self.conv_block(be, &x, &self.conv1, &self.norm1)?; // [1, C, T2, F2]
        let shape = be.shape(&x);
        let (c, t, f) = (shape[1], shape[2], shape[3]);
        let valid = valid1.div_ceil(2).min(t);
        let x = be.transpose(&be.transpose(&x, 1, 2)?, 2, 3)?; // [1, T, F, C]
        let x = be.reshape(&x, &[1, t, f * c])?;
        let mut x = self.input_proj.forward(be, &x)?;

        let bias_index = self.relative_index(be, t)?;
        let (keep, neg) = self.local_mask(be, t, valid)?;
        for l in &self.layers {
            x = self.feed_forward(be, &l.ff1, &x)?;
            let residual = x.clone();
            let h = be.rms_norm(&x, Some(&l.norm_pre_attn), eps)?;
            let h = self.attention(be, &l.attn, &h, &bias_index, &keep, &neg)?;
            let h = be.rms_norm(&h, Some(&l.norm_post_attn), eps)?;
            x = be.add(&h, &residual)?;
            x = self.light_conv(be, &l.lconv, &x)?;
            x = self.feed_forward(be, &l.ff2, &x)?;
            x = be.rms_norm(&x, Some(&l.norm_out), eps)?;
        }
        let x = self.output_proj.forward(be, &x)?;
        let x = be.reshape(
            &be.narrow(&x, 1, 0, valid)?,
            &[valid, self.cfg.output_proj_dims],
        )?;
        let x = be.rms_norm(&x, None, eps)?;
        self.embed_projection.forward(be, &x)
    }

    /// 3×3 stride-2 conv, LayerNorm over channels, ReLU.
    fn conv_block(
        &self,
        be: &B,
        x: &B::Tensor,
        w: &B::Tensor,
        norm: &B::Tensor,
    ) -> Result<B::Tensor> {
        let x = be.conv2d(x, w, 2, 1)?; // [1, C, T, F]
        let x = be.transpose(&be.transpose(&x, 1, 2)?, 2, 3)?; // [1, T, F, C]
        let x = be.layer_norm(&x, norm, self.cfg.rms_norm_eps)?;
        let x = be.transpose(&be.transpose(&x, 2, 3)?, 1, 2)?; // [1, C, T, F]
        be.relu(&x)
    }

    fn feed_forward(
        &self,
        be: &B,
        ff: &FeedForward<B::Tensor>,
        x: &B::Tensor,
    ) -> Result<B::Tensor> {
        let eps = self.cfg.rms_norm_eps;
        let h = be.rms_norm(x, Some(&ff.pre_norm), eps)?;
        let h = be.silu(&ff.ffw1.forward(be, &h)?)?;
        let h = ff.ffw2.forward(be, &h)?;
        let h = be.rms_norm(&h, Some(&ff.post_norm), eps)?;
        be.add(&be.affine(&h, self.cfg.residual_weight, 0.0)?, x)
    }

    fn light_conv(&self, be: &B, lc: &LightConv<B::Tensor>, x: &B::Tensor) -> Result<B::Tensor> {
        let eps = self.cfg.rms_norm_eps;
        let h = be.rms_norm(x, Some(&lc.pre_norm), eps)?;
        let h = lc.linear_start.forward(be, &h)?; // [1, T, 2C]
        let shape = be.shape(&h);
        let (t, c2) = (shape[1], shape[2]);
        let c = c2 / 2;
        // GLU: first half × sigmoid(second half).
        let h = be.mul(
            &be.narrow(&h, 2, 0, c)?,
            &be.sigmoid(&be.narrow(&h, 2, c, c)?)?,
        )?;
        // Causal depthwise conv: out[t] = Σ_k w[k] · h[t - (K-1) + k].
        let taps = lc.taps.len();
        let zeros = be.cast(
            &be.f32_tensor(&vec![0.0; (taps - 1) * c], &[1, taps - 1, c])?,
            be.precision(),
        )?;
        let padded = be.cat(&[zeros, h], 1)?;
        let mut acc: Option<B::Tensor> = None;
        for (k, tap) in lc.taps.iter().enumerate() {
            let term = be.mul(&be.narrow(&padded, 1, k, t)?, tap)?;
            acc = Some(match acc {
                None => term,
                Some(a) => be.add(&a, &term)?,
            });
        }
        let h = acc.expect("kernel has taps");
        let h = be.rms_norm(&h, Some(&lc.conv_norm), eps)?;
        let h = be.silu(&h)?;
        let h = lc.linear_end.forward(be, &h)?;
        be.add(&h, x)
    }

    /// Gather indices `[1, heads, T, T]` into the relative table: row
    /// `context/2 - (t - s)` (clamped; out-of-window entries are masked).
    fn relative_index(&self, be: &B, t: usize) -> Result<B::Tensor> {
        let heads = self.cfg.num_attention_heads;
        let max = (be.shape(&self.rel_pos)[0] - 1) as i64;
        let mut idx = Vec::with_capacity(t * t);
        for q in 0..t as i64 {
            for k in 0..t as i64 {
                idx.push((max - (q - k)).clamp(0, max) as u32);
            }
        }
        let mut all = Vec::with_capacity(heads * t * t);
        for _ in 0..heads {
            all.extend_from_slice(&idx);
        }
        be.u32_tensor(&all, &[1, heads, t, t])
    }

    /// Multiplicative keep mask and additive fill for `masked_fill(…,
    /// invalid_logits_value)`: a query at `q` sees real keys `k` with
    /// `0 <= q - k < attention_context_left - 1`.
    fn local_mask(&self, be: &B, t: usize, valid: usize) -> Result<(B::Tensor, B::Tensor)> {
        let left = self.cfg.attention_context_left.saturating_sub(1);
        let fill = self.cfg.attention_invalid_logits_value as f32;
        let mut keep = Vec::with_capacity(t * t);
        let mut neg = Vec::with_capacity(t * t);
        for q in 0..t {
            for k in 0..t {
                let ok = k < valid && k <= q && q - k < left;
                keep.push(if ok { 1.0 } else { 0.0 });
                neg.push(if ok { 0.0 } else { fill });
            }
        }
        Ok((
            be.f32_tensor(&keep, &[1, 1, t, t])?,
            be.f32_tensor(&neg, &[1, 1, t, t])?,
        ))
    }

    fn attention(
        &self,
        be: &B,
        a: &Attention<B::Tensor>,
        x: &B::Tensor,
        bias_index: &B::Tensor,
        keep: &B::Tensor,
        neg: &B::Tensor,
    ) -> Result<B::Tensor> {
        let heads = self.cfg.num_attention_heads;
        let hidden = self.cfg.hidden_size;
        let hd = hidden / heads;
        let t = be.shape(x)[1];
        let split = |y: B::Tensor| -> Result<B::Tensor> {
            let y = be.cast(&y, Precision::F32)?;
            be.transpose(&be.reshape(&y, &[1, t, heads, hd])?, 1, 2) // [1, H, T, D]
        };
        let q = split(a.q_proj.forward(be, x)?)?;
        let k = split(a.k_proj.forward(be, x)?)?;
        let v = split(a.v_proj.forward(be, x)?)?;
        let q = be.mul(&q, &a.q_scale)?;
        let k_scale = (1.0 + std::f64::consts::E).ln() / std::f64::consts::LN_2;
        let k = be.affine(&k, k_scale, 0.0)?;

        // Relative keys [n_pos, H, D] → [1, H, D, n_pos].
        let n_pos = be.shape(&self.rel_pos)[0];
        let rel = a.relative_k_proj.forward(be, &self.rel_pos)?;
        let rel = be.cast(&be.reshape(&rel, &[n_pos, heads, hd])?, Precision::F32)?;
        let rel = be.reshape(
            &be.transpose(&be.transpose(&rel, 0, 1)?, 1, 2)?,
            &[1, heads, hd, n_pos],
        )?;

        let ac = be.matmul(&q, &be.transpose(&k, 2, 3)?)?; // [1, H, T, T]
        let bd = be.gather_last(&be.matmul(&q, &rel)?, bias_index)?; // [1, H, T, T]
        let cap = self.cfg.attention_logit_cap;
        let logits = be.affine(
            &be.tanh(&be.affine(&be.add(&ac, &bd)?, 1.0 / cap, 0.0)?)?,
            cap,
            0.0,
        )?;
        let logits = be.add(&be.mul(&logits, keep)?, neg)?;
        let probs = be.softmax_last(&logits)?;
        let out = be.matmul(&probs, &v)?; // [1, H, T, D]
        let out = be.reshape(&be.transpose(&out, 1, 2)?, &[1, t, hidden])?;
        a.post.forward(be, &be.cast(&out, be.precision())?)
    }
}
