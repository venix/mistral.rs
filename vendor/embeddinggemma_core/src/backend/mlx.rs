//! [`Backend`] over Apple MLX (`mlx-rs` 0.32): the Metal 4 stack that drives
//! the GPU Neural Accelerators of M5-class Apple Silicon (macOS 26.2+), and
//! plain Metal on earlier chips. Build with `--features mlx` on macOS.
//!
//! MLX is lazy: operations build a graph that is evaluated when values are
//! read back ([`Backend::to_f32_vec`]). Arrays live in unified memory, so
//! there is no device transfer. Convolutions are NHWC in MLX; [`conv2d`]
//! converts from the PyTorch layout the model uses.
//!
//! [`conv2d`]: Backend::conv2d

use super::{AttentionMask, Backend};
use crate::device::Precision;
use crate::error::{EmbedError, Result};
use mlx_rs::{Array, Dtype, fast, nn, ops};
use std::collections::HashMap;
use std::path::PathBuf;

/// An MLX array.
///
/// `mlx_rs::Array` is `Send` but not `Sync`. The [`crate::Embedder`] runs
/// every forward pass under its compute lock, so the model's arrays are
/// never touched from two threads at once; this wrapper records that.
#[derive(Clone, Debug)]
pub struct MlxTensor(pub Array);

// SAFETY: see the type docs — all access is serialized by the embedder.
unsafe impl Sync for MlxTensor {}

pub struct MlxWeights {
    tensors: HashMap<String, Array>,
}

// SAFETY: the map is read only while loading, on one thread.
unsafe impl Sync for MlxWeights {}

pub struct MlxBackend {
    precision: Precision,
}

impl MlxBackend {
    pub fn new(precision: Precision) -> Result<Self> {
        Ok(Self { precision })
    }
}

fn err(e: impl std::fmt::Display) -> EmbedError {
    EmbedError::Backend(format!("mlx: {e}"))
}

fn dims(shape: &[usize]) -> Vec<i32> {
    shape.iter().map(|&d| d as i32).collect()
}

fn dtype(p: Precision) -> Dtype {
    match p {
        Precision::F32 => Dtype::Float32,
        Precision::BF16 => Dtype::Bfloat16,
        Precision::F16 => Dtype::Float16,
    }
}

fn t(a: Array) -> MlxTensor {
    MlxTensor(a)
}

fn scalar(x: f64, like: &Array) -> Result<Array> {
    Array::from_f32(x as f32)
        .as_dtype(like.dtype())
        .map_err(err)
}

impl Backend for MlxBackend {
    type Tensor = MlxTensor;
    type Weights = MlxWeights;

    fn device_name(&self) -> String {
        "mlx".into()
    }

    fn precision(&self) -> Precision {
        self.precision
    }

    fn open_weights(&self, paths: &[PathBuf]) -> Result<MlxWeights> {
        let mut tensors = HashMap::new();
        for p in paths {
            tensors.extend(Array::load_safetensors(p).map_err(err)?);
        }
        Ok(MlxWeights { tensors })
    }

    fn has_weight(&self, weights: &MlxWeights, name: &str) -> bool {
        weights.tensors.contains_key(name)
    }

    fn weight(&self, weights: &MlxWeights, name: &str, f32: bool) -> Result<MlxTensor> {
        let a = weights
            .tensors
            .get(name)
            .ok_or_else(|| EmbedError::Config(format!("checkpoint has no tensor `{name}`")))?;
        let target = if f32 {
            Dtype::Float32
        } else {
            dtype(self.precision)
        };
        Ok(t(a.as_dtype(target).map_err(err)?))
    }

    fn f32_tensor(&self, data: &[f32], shape: &[usize]) -> Result<MlxTensor> {
        Ok(t(Array::from_slice(data, &dims(shape))))
    }

    fn u32_tensor(&self, data: &[u32], shape: &[usize]) -> Result<MlxTensor> {
        Ok(t(Array::from_slice(data, &dims(shape))))
    }

    fn to_f32_vec(&self, x: &MlxTensor) -> Result<Vec<f32>> {
        let a = x.0.as_dtype(Dtype::Float32).map_err(err)?;
        // A contiguous copy so `as_slice` sees row-major data.
        let a = a.reshape(&[-1]).map_err(err)?;
        a.eval().map_err(err)?;
        Ok(a.as_slice::<f32>().to_vec())
    }

    fn shape(&self, x: &MlxTensor) -> Vec<usize> {
        x.0.shape().iter().map(|&d| d as usize).collect()
    }

    fn cast(&self, x: &MlxTensor, precision: Precision) -> Result<MlxTensor> {
        Ok(t(x.0.as_dtype(dtype(precision)).map_err(err)?))
    }

    fn reshape(&self, x: &MlxTensor, shape: &[usize]) -> Result<MlxTensor> {
        Ok(t(x.0.reshape(&dims(shape)).map_err(err)?))
    }

    fn transpose(&self, x: &MlxTensor, a: usize, b: usize) -> Result<MlxTensor> {
        Ok(t(x.0.swap_axes(a as i32, b as i32).map_err(err)?))
    }

    fn narrow(&self, x: &MlxTensor, dim: usize, start: usize, len: usize) -> Result<MlxTensor> {
        let idx: Vec<i32> = (start as i32..(start + len) as i32).collect();
        let idx = Array::from_slice(&idx, &[len as i32]);
        Ok(t(x.0.take_axis(&idx, dim as i32).map_err(err)?))
    }

    fn cat(&self, xs: &[MlxTensor], dim: usize) -> Result<MlxTensor> {
        let arrays: Vec<&Array> = xs.iter().map(|x| &x.0).collect();
        Ok(t(ops::concatenate_axis(&arrays, dim as i32).map_err(err)?))
    }

    fn add(&self, a: &MlxTensor, b: &MlxTensor) -> Result<MlxTensor> {
        Ok(t(a.0.add(&b.0).map_err(err)?))
    }

    fn sub(&self, a: &MlxTensor, b: &MlxTensor) -> Result<MlxTensor> {
        Ok(t(a.0.subtract(&b.0).map_err(err)?))
    }

    fn mul(&self, a: &MlxTensor, b: &MlxTensor) -> Result<MlxTensor> {
        Ok(t(a.0.multiply(&b.0).map_err(err)?))
    }

    fn affine(&self, x: &MlxTensor, mul: f64, add: f64) -> Result<MlxTensor> {
        let y = x.0.multiply(scalar(mul, &x.0)?).map_err(err)?;
        if add == 0.0 {
            return Ok(t(y));
        }
        Ok(t(y.add(scalar(add, &x.0)?).map_err(err)?))
    }

    fn clamp(&self, x: &MlxTensor, min: f64, max: f64) -> Result<MlxTensor> {
        let y = ops::maximum(&x.0, scalar(min, &x.0)?).map_err(err)?;
        Ok(t(ops::minimum(&y, scalar(max, &x.0)?).map_err(err)?))
    }

    fn tanh(&self, x: &MlxTensor) -> Result<MlxTensor> {
        Ok(t(ops::tanh(&x.0).map_err(err)?))
    }

    fn sigmoid(&self, x: &MlxTensor) -> Result<MlxTensor> {
        Ok(t(ops::sigmoid(&x.0).map_err(err)?))
    }

    fn relu(&self, x: &MlxTensor) -> Result<MlxTensor> {
        Ok(t(nn::relu(&x.0).map_err(err)?))
    }

    fn silu(&self, x: &MlxTensor) -> Result<MlxTensor> {
        Ok(t(nn::silu(&x.0).map_err(err)?))
    }

    fn gelu_tanh(&self, x: &MlxTensor) -> Result<MlxTensor> {
        Ok(t(nn::gelu_approximate(&x.0).map_err(err)?))
    }

    fn sum(&self, x: &MlxTensor, dim: usize, keepdim: bool) -> Result<MlxTensor> {
        Ok(t(x.0.sum_axis(dim as i32, keepdim).map_err(err)?))
    }

    fn softmax_last(&self, x: &MlxTensor) -> Result<MlxTensor> {
        Ok(t(ops::softmax_axis(&x.0, -1, true).map_err(err)?))
    }

    fn matmul(&self, a: &MlxTensor, b: &MlxTensor) -> Result<MlxTensor> {
        Ok(t(a.0.matmul(&b.0).map_err(err)?))
    }

    fn linear(&self, x: &MlxTensor, w: &MlxTensor, b: Option<&MlxTensor>) -> Result<MlxTensor> {
        let y = x.0.matmul(w.0.transpose().map_err(err)?).map_err(err)?;
        Ok(t(match b {
            Some(b) => y.add(&b.0).map_err(err)?,
            None => y,
        }))
    }

    fn embedding(&self, table: &MlxTensor, ids: &MlxTensor) -> Result<MlxTensor> {
        Ok(t(table.0.take_axis(&ids.0, 0).map_err(err)?))
    }

    fn rms_norm(&self, x: &MlxTensor, w: Option<&MlxTensor>, eps: f64) -> Result<MlxTensor> {
        let original = x.0.dtype();
        let x32 = x.0.as_dtype(Dtype::Float32).map_err(err)?;
        let w32 = match w {
            Some(w) => Some(w.0.as_dtype(Dtype::Float32).map_err(err)?),
            None => None,
        };
        let y = fast::rms_norm(&x32, w32.as_ref(), eps as f32).map_err(err)?;
        Ok(t(y.as_dtype(original).map_err(err)?))
    }

    fn layer_norm(&self, x: &MlxTensor, w: &MlxTensor, eps: f64) -> Result<MlxTensor> {
        let original = x.0.dtype();
        let x32 = x.0.as_dtype(Dtype::Float32).map_err(err)?;
        let w32 = w.0.as_dtype(Dtype::Float32).map_err(err)?;
        let y = fast::layer_norm(&x32, &w32, None::<&Array>, eps as f32).map_err(err)?;
        Ok(t(y.as_dtype(original).map_err(err)?))
    }

    fn rope(&self, x: &MlxTensor, cos: &MlxTensor, sin: &MlxTensor) -> Result<MlxTensor> {
        // Rotate-half with explicit tables (positions may be arbitrary, as
        // for the vision tower's 2-D grid), duplicating cos/sin to full width.
        let d = *x.0.shape().last().expect("rank ≥ 1") as usize;
        let half = d / 2;
        let x1 = self.narrow(x, x.0.ndim() - 1, 0, half)?;
        let x2 = self.narrow(x, x.0.ndim() - 1, half, half)?;
        let rot1 = self.sub(&self.mul(&x1, cos)?, &self.mul(&x2, sin)?)?;
        let rot2 = self.add(&self.mul(&x2, cos)?, &self.mul(&x1, sin)?)?;
        self.cat(&[rot1, rot2], x.0.ndim() - 1)
    }

    fn attention(
        &self,
        q: &MlxTensor,
        k: &MlxTensor,
        v: &MlxTensor,
        mask: &AttentionMask,
        scale: f64,
    ) -> Result<MlxTensor> {
        let shape = q.0.shape().to_vec(); // [b, h, s, d]
        let (b, h, s, d) = (shape[0], shape[1], shape[2], shape[3]);
        let additive = mask.additive(0, mask.seq_len, -1e9);
        let m = Array::from_slice(&additive, &[b, 1, s, s])
            .as_dtype(q.0.dtype())
            .map_err(err)?;
        let out = fast::scaled_dot_product_attention(
            &q.0,
            &k.0,
            &v.0,
            scale as f32,
            // `&Array` is the additive-mask form of `ScaledDotProductAttentionMask`.
            &m,
            None::<&Array>,
        )
        .map_err(err)?;
        let out = out.swap_axes(1, 2).map_err(err)?;
        Ok(t(out.reshape(&[b, s, h * d]).map_err(err)?))
    }

    fn gather_last(&self, x: &MlxTensor, idx: &MlxTensor) -> Result<MlxTensor> {
        Ok(t(x.0.take_along_axis(&idx.0, -1).map_err(err)?))
    }

    fn conv2d(
        &self,
        x: &MlxTensor,
        w: &MlxTensor,
        stride: usize,
        padding: usize,
    ) -> Result<MlxTensor> {
        // NCHW / OIHW → NHWC / OHWI, convolve, back to NCHW.
        let xn = x.0.transpose_axes(&[0, 2, 3, 1]).map_err(err)?;
        let wn = w.0.transpose_axes(&[0, 2, 3, 1]).map_err(err)?;
        let (s, p) = (stride as i32, padding as i32);
        let y =
            ops::conv2d(&xn, &wn, (s, s), (p, p), None::<(i32, i32)>, None::<i32>).map_err(err)?;
        Ok(t(y.transpose_axes(&[0, 3, 1, 2]).map_err(err)?))
    }

    fn eval(&self, x: &MlxTensor) -> Result<()> {
        x.0.eval().map_err(err)
    }
}
