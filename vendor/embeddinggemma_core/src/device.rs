//! Device and precision selection across CPU, CUDA, Metal and MLX builds.

use crate::error::{EmbedError, Result};
use serde::Serialize;
use std::fmt;
use std::str::FromStr;

/// Where to run. Parsed from `auto`, `cpu`, `cuda`, `cuda:1`, `metal`,
/// `metal:0`, `mlx`, or `gpu` (= auto but refuse CPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceRequest {
    /// CUDA if built with `cuda` and a GPU is present, then MLX if built with
    /// `mlx` (macOS), then candle's Metal if built with `metal`, otherwise CPU.
    #[default]
    Auto,
    Cpu,
    Cuda(usize),
    /// candle's Metal backend (any Apple GPU).
    Metal(usize),
    /// Apple MLX — the Metal 4 stack that drives the M5 GPU Neural
    /// Accelerators. Requires the `mlx` feature and macOS on Apple Silicon.
    Mlx,
    /// Any accelerator; error rather than silently fall back to CPU.
    Gpu,
}

impl FromStr for DeviceRequest {
    type Err = EmbedError;

    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        let (kind, ordinal) =
            match s.split_once(':') {
                Some((k, o)) => (
                    k.to_string(),
                    Some(o.parse::<usize>().map_err(|_| {
                        EmbedError::Device(format!("invalid device ordinal in `{s}`"))
                    })?),
                ),
                None => (s.clone(), None),
            };
        Ok(match kind.as_str() {
            "" | "auto" => Self::Auto,
            "cpu" => Self::Cpu,
            "cuda" | "nvidia" => Self::Cuda(ordinal.unwrap_or(0)),
            "metal" | "mps" => Self::Metal(ordinal.unwrap_or(0)),
            "mlx" | "apple" => Self::Mlx,
            "gpu" => Self::Gpu,
            other => {
                return Err(EmbedError::Device(format!(
                    "unknown device `{other}` (expected auto, cpu, cuda[:n], metal[:n], mlx or gpu)"
                )));
            }
        })
    }
}

impl fmt::Display for DeviceRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda(n) => write!(f, "cuda:{n}"),
            Self::Metal(n) => write!(f, "metal:{n}"),
            Self::Mlx => write!(f, "mlx"),
            Self::Gpu => write!(f, "gpu"),
        }
    }
}

/// Whether the MLX backend is compiled into this build.
pub const MLX_AVAILABLE: bool = cfg!(all(feature = "mlx", target_os = "macos"));

/// Backends compiled into this build.
pub fn compiled_backends() -> Vec<&'static str> {
    let mut out = vec!["cpu"];
    if cfg!(feature = "cuda") {
        out.push("cuda");
    }
    if cfg!(feature = "metal") {
        out.push("metal");
    }
    if MLX_AVAILABLE {
        out.push("mlx");
    }
    out
}

/// Compute precision of activations (and of dense weights).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    F32,
    BF16,
    /// Accepted, but the model card warns that float16 activations overflow
    /// (NaNs); prefer bf16 or f32.
    F16,
}

impl Precision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parse `f32` / `bf16` / `f16` (aliases: `float32`, `bfloat16`, `float16`, `half`).
pub fn parse_precision(s: &str) -> Result<Precision> {
    match s.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" | "float" => Ok(Precision::F32),
        "bf16" | "bfloat16" => Ok(Precision::BF16),
        "f16" | "float16" | "half" => Ok(Precision::F16),
        other => Err(EmbedError::Device(format!(
            "unknown dtype `{other}` (expected f32, bf16 or f16)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_device_requests() {
        assert_eq!(
            "auto".parse::<DeviceRequest>().unwrap(),
            DeviceRequest::Auto
        );
        assert_eq!("CPU".parse::<DeviceRequest>().unwrap(), DeviceRequest::Cpu);
        assert_eq!(
            "cuda".parse::<DeviceRequest>().unwrap(),
            DeviceRequest::Cuda(0)
        );
        assert_eq!(
            "cuda:2".parse::<DeviceRequest>().unwrap(),
            DeviceRequest::Cuda(2)
        );
        assert_eq!(
            "metal:1".parse::<DeviceRequest>().unwrap(),
            DeviceRequest::Metal(1)
        );
        assert_eq!("mlx".parse::<DeviceRequest>().unwrap(), DeviceRequest::Mlx);
        assert_eq!("gpu".parse::<DeviceRequest>().unwrap(), DeviceRequest::Gpu);
        assert!("tpu".parse::<DeviceRequest>().is_err());
        assert!("cuda:x".parse::<DeviceRequest>().is_err());
    }

    #[test]
    fn parses_precisions() {
        assert_eq!(parse_precision("bfloat16").unwrap(), Precision::BF16);
        assert_eq!(parse_precision("F32").unwrap(), Precision::F32);
        assert!(parse_precision("int8").is_err());
        assert!(compiled_backends().contains(&"cpu"));
    }
}
