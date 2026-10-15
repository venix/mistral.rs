//! [`Backend`] over candle: CPU, CUDA (`--features cuda`) and Metal
//! (`--features metal`).

use super::{AttentionMask, Backend};
use crate::device::{DeviceRequest, Precision};
use crate::error::{EmbedError, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{D, DType, Device, Module, Tensor};
use std::path::PathBuf;

pub struct CandleBackend {
    device: Device,
    precision: Precision,
}

impl CandleBackend {
    pub fn new(device: Device, precision: Precision) -> Self {
        Self { device, precision }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        dtype(self.precision)
    }
}

pub fn dtype(p: Precision) -> DType {
    match p {
        Precision::F32 => DType::F32,
        Precision::BF16 => DType::BF16,
        Precision::F16 => DType::F16,
    }
}

/// Resolve a request to a candle device (`Mlx` is not a candle device).
pub fn select_device(request: DeviceRequest) -> Result<Device> {
    match request {
        DeviceRequest::Cpu => Ok(Device::Cpu),
        DeviceRequest::Cuda(ordinal) => cuda(ordinal),
        DeviceRequest::Metal(ordinal) => metal(ordinal),
        DeviceRequest::Auto => Ok(first_accelerator().unwrap_or(Device::Cpu)),
        DeviceRequest::Gpu => first_accelerator().ok_or_else(|| {
            EmbedError::Device(format!(
                "no accelerator available (compiled backends: {})",
                crate::device::compiled_backends().join(", ")
            ))
        }),
        DeviceRequest::Mlx => Err(EmbedError::Device(
            "mlx is not a candle device (build with --features mlx on macOS)".into(),
        )),
    }
}

/// The first CUDA, then Metal device, if any is compiled in and present.
pub fn first_accelerator() -> Option<Device> {
    if cfg!(feature = "cuda")
        && let Ok(d) = cuda(0)
    {
        return Some(d);
    }
    if cfg!(feature = "metal")
        && let Ok(d) = metal(0)
    {
        return Some(d);
    }
    None
}

#[cfg(feature = "cuda")]
fn cuda(ordinal: usize) -> Result<Device> {
    Device::new_cuda(ordinal).map_err(|e| EmbedError::Device(format!("cuda:{ordinal}: {e}")))
}

#[cfg(not(feature = "cuda"))]
fn cuda(ordinal: usize) -> Result<Device> {
    Err(EmbedError::Device(format!(
        "cuda:{ordinal} requested but this binary was built without the `cuda` feature"
    )))
}

#[cfg(feature = "metal")]
fn metal(ordinal: usize) -> Result<Device> {
    Device::new_metal(ordinal).map_err(|e| EmbedError::Device(format!("metal:{ordinal}: {e}")))
}

#[cfg(not(feature = "metal"))]
fn metal(ordinal: usize) -> Result<Device> {
    Err(EmbedError::Device(format!(
        "metal:{ordinal} requested but this binary was built without the `metal` feature"
    )))
}

pub fn device_name(device: &Device) -> String {
    match device {
        Device::Cpu => "cpu".into(),
        Device::Cuda(_) => "cuda".into(),
        Device::Metal(_) => "metal".into(),
    }
}

/// Upper bound on f32 score elements materialised per attention chunk
/// (64M floats = 256 MiB). Long inputs are processed in query chunks so
/// memory stays flat instead of growing with `seq_len²`.
const ATTENTION_SCORE_BUDGET: usize = 1 << 26;

fn query_chunk(batch: usize, heads: usize, seq_len: usize) -> usize {
    let seq_len = seq_len.max(1);
    let per_row = batch * heads * seq_len;
    (ATTENTION_SCORE_BUDGET / per_row.max(1))
        .max(16)
        .min(seq_len)
}

/// Expand `[b, kv_heads, s, d]` to `[b, kv_heads * n_rep, s, d]` for GQA.
fn repeat_kv(x: &Tensor, n_rep: usize) -> candle_core::Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, h, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h, n_rep, s, d))?
        .reshape((b, h * n_rep, s, d))
}

/// Additive f32 mask `[b, 1, q_len, k_len]` for query rows
/// `q_start..q_start + q_len`, built on the device: `0` where
/// [`AttentionMask::allowed`], the most negative finite f32 where not (a
/// fully masked row softmaxes to uniform instead of NaN, like `transformers`
/// with `finfo.min`).
pub(crate) fn device_mask(
    mask: &AttentionMask,
    q_start: usize,
    q_len: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let k_len = mask.seq_len;
    let b = mask.lengths.len();
    let q_pos =
        Tensor::arange(q_start as u32, (q_start + q_len) as u32, device)?.reshape((q_len, 1))?;
    let k_pos = Tensor::arange(0u32, k_len as u32, device)?.reshape((1, k_len))?;
    let mut allowed = Tensor::ones((q_len, k_len), DType::U8, device)?;
    if let Some(w) = mask.window {
        // q - k <= w  <=>  k + w >= q
        let k_plus_w = Tensor::arange(w as u32, (w + k_len) as u32, device)?.reshape((1, k_len))?;
        allowed = allowed.mul(&k_plus_w.broadcast_ge(&q_pos)?)?;
        // k - q <= w  <=>  k <= q + w
        let q_plus_w = Tensor::arange((q_start + w) as u32, (q_start + q_len + w) as u32, device)?
            .reshape((q_len, 1))?;
        allowed = allowed.mul(&k_pos.broadcast_le(&q_plus_w)?)?;
    }
    let lengths: Vec<u32> = mask.lengths.iter().map(|&l| l as u32).collect();
    let lengths = Tensor::from_vec(lengths, (b, 1, 1, 1), device)?;
    let key_valid = k_pos.reshape((1, 1, 1, k_len))?.broadcast_lt(&lengths)?;
    let allowed = allowed
        .reshape((1, 1, q_len, k_len))?
        .broadcast_mul(&key_valid)?;
    let zeros = Tensor::zeros((b, 1, q_len, k_len), DType::F32, device)?;
    let neg = Tensor::full(f32::MIN, (b, 1, q_len, k_len), device)?;
    allowed.where_cond(&zeros, &neg)
}

/// Exact attention in query chunks (see [`ATTENTION_SCORE_BUDGET`]).
pub(crate) fn attention_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &AttentionMask,
    scale: f64,
    chunk: usize,
) -> candle_core::Result<Tensor> {
    let (b, heads, s, d) = q.dims4()?;
    let kv_heads = k.dim(1)?;
    let n_rep = heads / kv_heads;
    let dtype = q.dtype();
    let device = q.device();

    let k = repeat_kv(k, n_rep)?;
    let v = repeat_kv(v, n_rep)?.contiguous()?;
    let k_t = k.transpose(2, 3)?.contiguous()?; // [b, heads, d, s]

    // No padding and no window (a single image, equal-length rows): the
    // mask would be all zeros, so skip building and adding it.
    let unmasked = mask.window.is_none() && mask.lengths.iter().all(|&l| l >= s);
    let chunk = chunk.clamp(1, s.max(1));
    let mut outputs = Vec::with_capacity(s.div_ceil(chunk));
    let mut start = 0;
    while start < s {
        let len = chunk.min(s - start);
        let q_chunk = q.narrow(2, start, len)?.contiguous()?;
        let scores = q_chunk.matmul(&k_t)?; // [b, heads, len, s]
        let scores = scores.to_dtype(DType::F32)?.affine(scale, 0.0)?;
        let scores = if unmasked {
            scores
        } else {
            scores.broadcast_add(&device_mask(mask, start, len, device)?)?
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(dtype)?;
        outputs.push(probs.matmul(&v)?); // [b, heads, len, d]
        start += len;
    }
    let out = if outputs.len() == 1 {
        outputs.pop().expect("one chunk")
    } else {
        Tensor::cat(&outputs, 2)?
    };
    out.transpose(1, 2)?.reshape((b, s, heads * d))
}

/// Opened safetensors shards; tensors are read on demand.
pub struct CandleWeights {
    st: MmapedSafetensors,
}

impl Backend for CandleBackend {
    type Tensor = Tensor;
    type Weights = CandleWeights;

    fn device_name(&self) -> String {
        device_name(&self.device)
    }

    fn precision(&self) -> Precision {
        self.precision
    }

    fn open_weights(&self, paths: &[PathBuf]) -> Result<CandleWeights> {
        // SAFETY: the safetensors files are memory-mapped read-only and are
        // not modified while the model lives.
        let st = unsafe { MmapedSafetensors::multi(paths)? };
        Ok(CandleWeights { st })
    }

    fn has_weight(&self, weights: &CandleWeights, name: &str) -> bool {
        weights.st.get(name).is_ok()
    }

    fn weight(&self, weights: &CandleWeights, name: &str, f32: bool) -> Result<Tensor> {
        let t = weights
            .st
            .load(name, &self.device)
            .map_err(|e| EmbedError::Config(format!("checkpoint tensor `{name}`: {e}")))?;
        let dtype = if f32 { DType::F32 } else { self.dtype() };
        Ok(t.to_dtype(dtype)?)
    }

    fn f32_tensor(&self, data: &[f32], shape: &[usize]) -> Result<Tensor> {
        Ok(Tensor::from_slice(data, shape, &self.device)?)
    }

    fn u32_tensor(&self, data: &[u32], shape: &[usize]) -> Result<Tensor> {
        Ok(Tensor::from_slice(data, shape, &self.device)?)
    }

    fn to_f32_vec(&self, x: &Tensor) -> Result<Vec<f32>> {
        Ok(x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
    }

    fn shape(&self, x: &Tensor) -> Vec<usize> {
        x.dims().to_vec()
    }

    fn cast(&self, x: &Tensor, precision: Precision) -> Result<Tensor> {
        Ok(x.to_dtype(dtype(precision))?)
    }

    fn reshape(&self, x: &Tensor, shape: &[usize]) -> Result<Tensor> {
        Ok(x.reshape(shape)?)
    }

    fn transpose(&self, x: &Tensor, a: usize, b: usize) -> Result<Tensor> {
        Ok(x.transpose(a, b)?)
    }

    fn narrow(&self, x: &Tensor, dim: usize, start: usize, len: usize) -> Result<Tensor> {
        Ok(x.narrow(dim, start, len)?)
    }

    fn cat(&self, xs: &[Tensor], dim: usize) -> Result<Tensor> {
        Ok(Tensor::cat(xs, dim)?)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Ok(a.broadcast_add(b)?)
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Ok(a.broadcast_sub(b)?)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Ok(a.broadcast_mul(b)?)
    }

    fn affine(&self, x: &Tensor, mul: f64, add: f64) -> Result<Tensor> {
        Ok(x.affine(mul, add)?)
    }

    fn clamp(&self, x: &Tensor, min: f64, max: f64) -> Result<Tensor> {
        Ok(x.clamp(min, max)?)
    }

    fn tanh(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.tanh()?)
    }

    fn sigmoid(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::sigmoid(x)?)
    }

    fn relu(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.relu()?)
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.silu()?)
    }

    fn gelu_tanh(&self, x: &Tensor) -> Result<Tensor> {
        // candle's `gelu` is the tanh approximation (= gelu_pytorch_tanh).
        Ok(x.gelu()?)
    }

    fn sum(&self, x: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
        Ok(if keepdim {
            x.sum_keepdim(dim)?
        } else {
            x.sum(dim)?
        })
    }

    fn softmax_last(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::softmax_last_dim(&x.contiguous()?)?)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Ok(a.contiguous()?.broadcast_matmul(&b.contiguous()?)?)
    }

    fn linear(&self, x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
        Ok(candle_nn::Linear::new(w.clone(), b.cloned()).forward(x)?)
    }

    fn embedding(&self, table: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let hidden = table.dim(1)?;
        let mut shape = ids.dims().to_vec();
        shape.push(hidden);
        Ok(table.index_select(&ids.flatten_all()?, 0)?.reshape(shape)?)
    }

    fn rms_norm(&self, x: &Tensor, w: Option<&Tensor>, eps: f64) -> Result<Tensor> {
        let dtype = x.dtype();
        let x32 = x.to_dtype(DType::F32)?.contiguous()?;
        let alpha = match w {
            Some(w) => w.to_dtype(DType::F32)?,
            None => Tensor::ones(x32.dim(D::Minus1)?, DType::F32, x.device())?,
        };
        let y = candle_nn::ops::rms_norm(&x32, &alpha, eps as f32)?;
        Ok(y.to_dtype(dtype)?)
    }

    fn layer_norm(&self, x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
        let dtype = x.dtype();
        let x32 = x.to_dtype(DType::F32)?.contiguous()?;
        let alpha = w.to_dtype(DType::F32)?;
        let beta = alpha.zeros_like()?;
        let y = candle_nn::ops::layer_norm(&x32, &alpha, &beta, eps as f32)?;
        Ok(y.to_dtype(dtype)?)
    }

    fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::rotary_emb::rope(
            &x.contiguous()?,
            &cos.contiguous()?,
            &sin.contiguous()?,
        )?)
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &AttentionMask,
        scale: f64,
    ) -> Result<Tensor> {
        let (b, heads, s, _) = q.dims4()?;
        Ok(attention_chunked(
            q,
            k,
            v,
            mask,
            scale,
            query_chunk(b, heads, s),
        )?)
    }

    fn gather_last(&self, x: &Tensor, idx: &Tensor) -> Result<Tensor> {
        let last = x.rank() - 1;
        Ok(x.contiguous()?.gather(&idx.contiguous()?, last)?)
    }

    fn conv2d(&self, x: &Tensor, w: &Tensor, stride: usize, padding: usize) -> Result<Tensor> {
        Ok(x.contiguous()?.conv2d(w, padding, stride, 1, 1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> CandleBackend {
        CandleBackend::new(Device::Cpu, Precision::F32)
    }

    #[test]
    fn device_mask_matches_the_reference_rule() {
        let mask = AttentionMask {
            lengths: vec![6, 3],
            seq_len: 6,
            window: Some(2),
        };
        let m = device_mask(&mask, 1, 4, &Device::Cpu).unwrap();
        let got: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        let want = mask.additive(1, 4, f32::MIN);
        assert_eq!(got, want);
    }

    #[test]
    fn chunked_attention_matches_naive() {
        let dev = Device::Cpu;
        let (b, h, s, d) = (2, 4, 37, 8);
        let q = Tensor::randn(0f32, 1.0, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1.0, (b, 2, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1.0, (b, 2, s, d), &dev).unwrap();
        let mask = AttentionMask {
            lengths: vec![37, 20],
            seq_len: s,
            window: Some(5),
        };
        let out = attention_chunked(&q, &k, &v, &mask, 0.7, 6).unwrap();
        assert_eq!(out.dims(), &[b, s, h * d]);
        let qv: Vec<f32> = q.flatten_all().unwrap().to_vec1().unwrap();
        let kv: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
        let vv: Vec<f32> = v.flatten_all().unwrap().to_vec1().unwrap();
        let ov: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for bi in 0..b {
            for hi in 0..h {
                let kh = hi / 2;
                for qi in 0..mask.lengths[bi] {
                    let keys: Vec<usize> = (0..s).filter(|&ki| mask.allowed(bi, qi, ki)).collect();
                    let scores: Vec<f32> = keys
                        .iter()
                        .map(|&ki| {
                            (0..d)
                                .map(|di| {
                                    qv[((bi * h + hi) * s + qi) * d + di]
                                        * kv[((bi * 2 + kh) * s + ki) * d + di]
                                })
                                .sum::<f32>()
                                * 0.7
                        })
                        .collect();
                    let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let e: Vec<f32> = scores.iter().map(|x| (x - m).exp()).collect();
                    let z: f32 = e.iter().sum();
                    for di in 0..d {
                        let want: f32 = keys
                            .iter()
                            .zip(&e)
                            .map(|(&ki, w)| w / z * vv[((bi * 2 + kh) * s + ki) * d + di])
                            .sum();
                        let got = ov[(bi * s + qi) * (h * d) + hi * d + di];
                        assert!((got - want).abs() < 1e-4, "{got} vs {want}");
                    }
                }
            }
        }
    }

    #[test]
    fn primitives_follow_the_reference_formulas() {
        let be = backend();
        let x = be.f32_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let y = be
            .to_f32_vec(&be.rms_norm(&x, None, 1e-6).unwrap())
            .unwrap();
        let rms = (30.0f32 / 4.0 + 1e-6).sqrt();
        for (a, b) in y.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            assert!((a - b / rms).abs() < 1e-6);
        }
        let idx = be.u32_tensor(&[2, 0, 1, 1], &[1, 4]).unwrap();
        assert_eq!(
            be.to_f32_vec(&be.gather_last(&x, &idx).unwrap()).unwrap(),
            vec![3.0, 1.0, 2.0, 2.0]
        );
        let table = be
            .f32_tensor(&[0., 1., 10., 11., 20., 21.], &[3, 2])
            .unwrap();
        let ids = be.u32_tensor(&[2, 0], &[1, 2]).unwrap();
        let e = be.embedding(&table, &ids).unwrap();
        assert_eq!(be.shape(&e), vec![1, 2, 2]);
        assert_eq!(be.to_f32_vec(&e).unwrap(), vec![20., 21., 0., 1.]);
    }
}
