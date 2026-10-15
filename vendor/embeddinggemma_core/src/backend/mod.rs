//! The tensor operations the EmbeddingGemma 2 towers are written against.
//!
//! The model code in [`crate::model`] is generic over [`Backend`], so the
//! text, vision and audio towers exist once and run on:
//!
//! - [`candle::CandleBackend`] — CPU, CUDA (`--features cuda`) and Metal
//!   (`--features metal`) through candle;
//! - `mlx::MlxBackend` — Apple MLX (`--features mlx`, macOS only), the Metal 4
//!   stack that drives the M5 GPU Neural Accelerators.
//!
//! The trait is deliberately small: plain tensor algebra plus a handful of
//! fused primitives (`rms_norm`, `rope`, `attention`) that each backend
//! implements with its own fast path. Shapes follow PyTorch conventions
//! (`[batch, heads, seq, head_dim]`, linear weights `[out, in]`, conv
//! weights `[out, in, kh, kw]`); a backend with a different native layout
//! (MLX convolutions are NHWC) converts internally.

pub mod candle;
#[cfg(all(feature = "mlx", target_os = "macos"))]
pub mod mlx;

use crate::device::Precision;
use crate::error::Result;
use std::path::PathBuf;

/// Which keys a query may attend to in a padded, bidirectional batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionMask {
    /// Real (unpadded) length of every row; rows are right-padded to `seq_len`.
    pub lengths: Vec<usize>,
    pub seq_len: usize,
    /// Bidirectional sliding window: a query at `q` sees keys with
    /// `|q - k| <= window` (inclusive, as `transformers`'
    /// `sliding_window_bidirectional_overlay`). `None` = every real key.
    pub window: Option<usize>,
}

impl AttentionMask {
    /// Whether query `q` of row `b` may attend to key `k`.
    pub fn allowed(&self, b: usize, q: usize, k: usize) -> bool {
        k < self.lengths[b] && self.window.is_none_or(|w| q.abs_diff(k) <= w)
    }

    /// Dense `[b, 1, q_len, seq_len]` additive mask for query rows
    /// `q_start..q_start + q_len`: `0.0` where allowed, `neg` where not.
    pub fn additive(&self, q_start: usize, q_len: usize, neg: f32) -> Vec<f32> {
        let s = self.seq_len;
        let mut out = Vec::with_capacity(self.lengths.len() * q_len * s);
        for b in 0..self.lengths.len() {
            for q in q_start..q_start + q_len {
                out.extend((0..s).map(|k| if self.allowed(b, q, k) { 0.0 } else { neg }));
            }
        }
        out
    }
}

/// A tensor backend. Every method takes `&self` (the backend carries the
/// device and compute precision); tensors are cheap-to-clone handles.
pub trait Backend: Send + Sync + 'static {
    type Tensor: Clone + Send + Sync + std::fmt::Debug;
    /// An opened checkpoint (memory-mapped safetensors).
    type Weights: Send + Sync;

    /// `cpu`, `cuda`, `metal` or `mlx`, for logs and `/v1/models`.
    fn device_name(&self) -> String;
    /// Activation precision.
    fn precision(&self) -> Precision;

    // ---- weights -------------------------------------------------------
    fn open_weights(&self, paths: &[PathBuf]) -> Result<Self::Weights>;
    fn has_weight(&self, weights: &Self::Weights, name: &str) -> bool;
    /// Load one parameter in the compute precision, or in f32 when `f32`.
    fn weight(&self, weights: &Self::Weights, name: &str, f32: bool) -> Result<Self::Tensor>;

    // ---- host transfer -------------------------------------------------
    /// An f32 tensor from host data.
    fn f32_tensor(&self, data: &[f32], shape: &[usize]) -> Result<Self::Tensor>;
    /// A u32 tensor (token ids, gather indices) from host data.
    fn u32_tensor(&self, data: &[u32], shape: &[usize]) -> Result<Self::Tensor>;
    /// Flattened values, converted to f32.
    fn to_f32_vec(&self, x: &Self::Tensor) -> Result<Vec<f32>>;
    fn shape(&self, x: &Self::Tensor) -> Vec<usize>;
    fn cast(&self, x: &Self::Tensor, precision: Precision) -> Result<Self::Tensor>;

    // ---- layout --------------------------------------------------------
    fn reshape(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor>;
    fn transpose(&self, x: &Self::Tensor, a: usize, b: usize) -> Result<Self::Tensor>;
    fn narrow(
        &self,
        x: &Self::Tensor,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<Self::Tensor>;
    fn cat(&self, xs: &[Self::Tensor], dim: usize) -> Result<Self::Tensor>;

    // ---- elementwise (numpy broadcasting) -------------------------------
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    fn sub(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    /// `x * mul + add`.
    fn affine(&self, x: &Self::Tensor, mul: f64, add: f64) -> Result<Self::Tensor>;
    fn clamp(&self, x: &Self::Tensor, min: f64, max: f64) -> Result<Self::Tensor>;
    fn tanh(&self, x: &Self::Tensor) -> Result<Self::Tensor>;
    fn sigmoid(&self, x: &Self::Tensor) -> Result<Self::Tensor>;
    fn relu(&self, x: &Self::Tensor) -> Result<Self::Tensor>;
    fn silu(&self, x: &Self::Tensor) -> Result<Self::Tensor>;
    /// `gelu_pytorch_tanh`.
    fn gelu_tanh(&self, x: &Self::Tensor) -> Result<Self::Tensor>;

    // ---- reductions / products -----------------------------------------
    fn sum(&self, x: &Self::Tensor, dim: usize, keepdim: bool) -> Result<Self::Tensor>;
    /// Softmax over the last dimension (computed in f32 by the caller's
    /// choice of input precision).
    fn softmax_last(&self, x: &Self::Tensor) -> Result<Self::Tensor>;
    /// Batched matrix product over the last two dimensions, broadcasting the
    /// leading ones.
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
    /// `x @ w.T (+ b)` with `w: [out, in]`.
    fn linear(
        &self,
        x: &Self::Tensor,
        w: &Self::Tensor,
        b: Option<&Self::Tensor>,
    ) -> Result<Self::Tensor>;

    // ---- model primitives ----------------------------------------------
    /// Rows of `table [vocab, hidden]` for `ids` (u32, any shape).
    fn embedding(&self, table: &Self::Tensor, ids: &Self::Tensor) -> Result<Self::Tensor>;
    /// `x * (mean(x²) + eps)^-½ (* w)` over the last dimension, computed in
    /// f32 and returned in `x`'s precision (Gemma 4's RMSNorm — **not**
    /// Gemma 2/3's `1 + w`).
    fn rms_norm(
        &self,
        x: &Self::Tensor,
        w: Option<&Self::Tensor>,
        eps: f64,
    ) -> Result<Self::Tensor>;
    /// LayerNorm over the last dimension with a scale and no bias.
    fn layer_norm(&self, x: &Self::Tensor, w: &Self::Tensor, eps: f64) -> Result<Self::Tensor>;
    /// Rotate-half RoPE: `x [b, h, s, d]`, `cos`/`sin` `[s, d/2]` in `x`'s
    /// precision (one row per position of `x`, so positions may be arbitrary).
    fn rope(
        &self,
        x: &Self::Tensor,
        cos: &Self::Tensor,
        sin: &Self::Tensor,
    ) -> Result<Self::Tensor>;
    /// Scaled dot-product attention with grouped KV heads: `q [b, h, s, d]`,
    /// `k`/`v` `[b, kv_h, s, d]` → `[b, s, h * d]`. Scores are soft-maxed in
    /// f32.
    fn attention(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        mask: &AttentionMask,
        scale: f64,
    ) -> Result<Self::Tensor>;
    /// `out[..., j] = x[..., idx[..., j]]` along the last dimension (`idx`
    /// u32, same shape as the output, same leading dims as `x`).
    fn gather_last(&self, x: &Self::Tensor, idx: &Self::Tensor) -> Result<Self::Tensor>;
    /// 2-D convolution, no bias: `x [n, c_in, h, w]`, `w [c_out, c_in, kh, kw]`.
    fn conv2d(
        &self,
        x: &Self::Tensor,
        w: &Self::Tensor,
        stride: usize,
        padding: usize,
    ) -> Result<Self::Tensor>;

    /// Force pending work (lazy backends); a no-op for eager ones.
    fn eval(&self, _x: &Self::Tensor) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_is_padded_and_windowed_inclusively() {
        let m = AttentionMask {
            lengths: vec![4, 2],
            seq_len: 4,
            window: Some(1),
        };
        assert!(m.allowed(0, 0, 1) && !m.allowed(0, 0, 2));
        assert!(m.allowed(0, 2, 1) && m.allowed(0, 2, 3));
        assert!(!m.allowed(1, 0, 2), "padding key");
        let add = m.additive(1, 1, -1.0);
        assert_eq!(add, vec![0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, -1.0]);
    }
}
