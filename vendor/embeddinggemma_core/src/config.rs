//! `config.json` (`EmbeddingGemma2Config`) and `processor_config.json`.

use crate::error::{EmbedError, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

/// The composite model config.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub model_type: String,
    pub text_config: TextConfig,
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
    #[serde(default)]
    pub audio_config: Option<AudioConfig>,
    #[serde(default = "default_image_token")]
    pub image_token_id: u32,
    #[serde(default = "default_video_token")]
    pub video_token_id: u32,
    #[serde(default = "default_audio_token")]
    pub audio_token_id: u32,
    #[serde(default = "default_boi")]
    pub boi_token_id: u32,
    #[serde(default = "default_eoi")]
    pub eoi_token_id: u32,
    #[serde(default = "default_boa")]
    pub boa_token_id: u32,
    /// Spelled `eoa_token_index` in the published config.
    #[serde(default = "default_eoa", alias = "eoa_token_index")]
    pub eoa_token_id: u32,
}

fn default_image_token() -> u32 {
    258_880
}
fn default_audio_token() -> u32 {
    258_881
}
fn default_video_token() -> u32 {
    258_884
}
fn default_boi() -> u32 {
    255_999
}
fn default_eoi() -> u32 {
    258_882
}
fn default_boa() -> u32 {
    256_000
}
fn default_eoa() -> u32 {
    258_883
}

impl ModelConfig {
    pub fn parse(json: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(json)?;
        if !cfg.model_type.is_empty() && cfg.model_type != "embedding_gemma2" {
            return Err(EmbedError::Config(format!(
                "unsupported model_type `{}` (expected embedding_gemma2)",
                cfg.model_type
            )));
        }
        let t = &cfg.text_config;
        if t.layer_types.len() != t.num_hidden_layers {
            return Err(EmbedError::Config(format!(
                "layer_types has {} entries for {} layers",
                t.layer_types.len(),
                t.num_hidden_layers
            )));
        }
        Ok(cfg)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    /// Output width of `embedding_projection` (768).
    pub embedding_dim: usize,
    pub hidden_size_per_layer_input: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
    /// Overrides for some layers, keyed by zero-padded layer index (`"05"`).
    #[serde(default)]
    pub per_layer_config: BTreeMap<String, LayerOverride>,
    pub rope_parameters: BTreeMap<String, RopeParameters>,
    #[serde(default)]
    pub hidden_activation: Option<String>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayerOverride {
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn default_eps() -> f64 {
    1e-6
}

/// Resolved attention shape of one text layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextLayerSpec {
    pub sliding: bool,
    pub head_dim: usize,
    pub kv_heads: usize,
    pub rope_theta: f64,
}

impl TextConfig {
    /// Per-layer attention shape: `per_layer_config` entries override the
    /// defaults, RoPE comes from `rope_parameters[layer_type]`.
    pub fn layer_specs(&self) -> Result<Vec<TextLayerSpec>> {
        self.layer_types
            .iter()
            .enumerate()
            .map(|(i, kind)| {
                let o = self
                    .per_layer_config
                    .iter()
                    .find(|(k, _)| k.parse::<usize>().ok() == Some(i))
                    .map(|(_, v)| v);
                let rope = self.rope_parameters.get(kind).ok_or_else(|| {
                    EmbedError::Config(format!("no rope_parameters for layer type `{kind}`"))
                })?;
                if let Some(t) = &rope.rope_type
                    && t != "default"
                {
                    return Err(EmbedError::Config(format!(
                        "unsupported text rope_type `{t}` (only default)"
                    )));
                }
                Ok(TextLayerSpec {
                    sliding: kind == "sliding_attention",
                    head_dim: o.and_then(|o| o.head_dim).unwrap_or(self.head_dim),
                    kv_heads: o
                        .and_then(|o| o.num_key_value_heads)
                        .unwrap_or(self.num_key_value_heads),
                    rope_theta: rope.rope_theta,
                })
            })
            .collect()
    }

    pub fn pad_id(&self) -> u32 {
        self.pad_token_id.unwrap_or(0)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub standardize: bool,
    #[serde(default)]
    pub use_clipped_linears: bool,
    #[serde(default)]
    pub hidden_activation: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub attention_chunk_size: usize,
    pub attention_context_left: usize,
    pub attention_context_right: usize,
    #[serde(default = "default_invalid_logits")]
    pub attention_invalid_logits_value: f64,
    pub attention_logit_cap: f64,
    pub conv_kernel_size: usize,
    #[serde(default)]
    pub hidden_act: Option<String>,
    pub output_proj_dims: usize,
    #[serde(default = "default_residual_weight")]
    pub residual_weight: f64,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    pub subsampling_conv_channels: Vec<usize>,
    #[serde(default)]
    pub use_clipped_linears: bool,
}

fn default_invalid_logits() -> f64 {
    -1e9
}
fn default_residual_weight() -> f64 {
    0.5
}

/// The parts of `processor_config.json` (or `preprocessor_config.json`) that
/// drive input preparation. Every field has the published default.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessorConfig {
    /// Soft tokens per image: one of 70, 140, 280, 560, 1120.
    pub image_soft_tokens: usize,
    /// Soft tokens per video frame.
    pub video_soft_tokens: usize,
    pub video_fps: f64,
    pub video_max_frames: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub sampling_rate: usize,
    pub feature_size: usize,
    pub frame_length: usize,
    pub hop_length: usize,
    pub fft_length: usize,
    pub min_frequency: f64,
    pub max_frequency: f64,
    pub mel_floor: f64,
    /// The feature extractor truncates waveforms to this many samples
    /// (480 000 = 30 s).
    pub max_audio_samples: usize,
    /// …and pads them to a multiple of this many samples.
    pub pad_to_multiple_of: usize,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            image_soft_tokens: 280,
            video_soft_tokens: 140,
            video_fps: 1.0,
            video_max_frames: 32,
            patch_size: 16,
            pooling_kernel_size: 3,
            sampling_rate: 16_000,
            feature_size: 128,
            frame_length: 320,
            hop_length: 160,
            fft_length: 512,
            min_frequency: 0.0,
            max_frequency: 8000.0,
            mel_floor: 1e-3,
            max_audio_samples: 480_000,
            pad_to_multiple_of: 128,
        }
    }
}

/// Image soft-token budgets the processor accepts.
pub const SUPPORTED_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];

impl ProcessorConfig {
    /// Read `processor_config.json`; missing keys keep their defaults.
    pub fn parse(json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json)?;
        let mut c = Self::default();
        let num = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_f64());
        let image = v.get("image_processor").unwrap_or(&v);
        if let Some(x) = num(image, "max_soft_tokens").or_else(|| num(&v, "image_seq_length")) {
            c.image_soft_tokens = x as usize;
        }
        if let Some(x) = num(image, "patch_size") {
            c.patch_size = x as usize;
        }
        if let Some(x) = num(image, "pooling_kernel_size") {
            c.pooling_kernel_size = x as usize;
        }
        if let Some(video) = v.get("video_processor") {
            if let Some(x) = num(video, "max_soft_tokens") {
                c.video_soft_tokens = x as usize;
            }
            if let Some(x) = num(video, "fps") {
                c.video_fps = x;
            }
            if let Some(x) = num(video, "max_frames") {
                c.video_max_frames = x as usize;
            }
        }
        let audio = v.get("feature_extractor").unwrap_or(&v);
        if let Some(x) = num(audio, "sampling_rate") {
            c.sampling_rate = x as usize;
        }
        if let Some(x) = num(audio, "feature_size") {
            c.feature_size = x as usize;
        }
        if let Some(x) = num(audio, "frame_length") {
            c.frame_length = x as usize;
        }
        if let Some(x) = num(audio, "hop_length") {
            c.hop_length = x as usize;
        }
        if let Some(x) = num(audio, "fft_length") {
            c.fft_length = x as usize;
        }
        if let Some(x) = num(audio, "min_frequency") {
            c.min_frequency = x;
        }
        if let Some(x) = num(audio, "max_frequency") {
            c.max_frequency = x;
        }
        if let Some(x) = num(audio, "mel_floor") {
            c.mel_floor = x;
        }
        for (key, what) in [("dither", "dither"), ("preemphasis", "preemphasis")] {
            if num(audio, key).is_some_and(|x| x != 0.0) {
                return Err(EmbedError::Config(format!(
                    "audio feature extractor uses {what}, which is not implemented"
                )));
            }
        }
        if !SUPPORTED_SOFT_TOKENS.contains(&c.image_soft_tokens) {
            return Err(EmbedError::Config(format!(
                "image max_soft_tokens {} is not one of {SUPPORTED_SOFT_TOKENS:?}",
                c.image_soft_tokens
            )));
        }
        Ok(c)
    }
}

/// Named prompts from `config_sentence_transformers.json` (`"query"`,
/// `"Retrieval-document"`, …), keyed by name.
pub fn sentence_transformers_prompts(json: &str) -> Result<BTreeMap<String, String>> {
    #[derive(Deserialize)]
    struct St {
        #[serde(default)]
        prompts: BTreeMap<String, String>,
    }
    Ok(serde_json::from_str::<St>(json)?.prompts)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{"model_type":"embedding_gemma2","eoa_token_index":7,
        "text_config":{"hidden_size":8,"num_hidden_layers":3,"num_attention_heads":2,"num_key_value_heads":2,
        "head_dim":4,"intermediate_size":16,"vocab_size":10,"embedding_dim":12,"hidden_size_per_layer_input":8,
        "sliding_window":4,"layer_types":["sliding_attention","sliding_attention","full_attention"],
        "per_layer_config":{"02":{"head_dim":8,"num_key_value_heads":1}},
        "rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},
                           "sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}}}"#;

    #[test]
    fn resolves_per_layer_overrides() {
        let cfg = ModelConfig::parse(CONFIG).unwrap();
        assert_eq!(cfg.eoa_token_id, 7);
        assert_eq!(cfg.image_token_id, 258_880);
        let specs = cfg.text_config.layer_specs().unwrap();
        assert_eq!(
            specs[0],
            TextLayerSpec {
                sliding: true,
                head_dim: 4,
                kv_heads: 2,
                rope_theta: 10_000.0
            }
        );
        assert_eq!(
            specs[2],
            TextLayerSpec {
                sliding: false,
                head_dim: 8,
                kv_heads: 1,
                rope_theta: 1_000_000.0
            }
        );
    }

    #[test]
    fn reads_the_processor_config() {
        let json = r#"{"image_seq_length":280,"image_processor":{"max_soft_tokens":560,"patch_size":16},
            "video_processor":{"fps":2,"max_frames":16,"max_soft_tokens":70},
            "feature_extractor":{"sampling_rate":16000,"mel_floor":0.001,"dither":0.0}}"#;
        let c = ProcessorConfig::parse(json).unwrap();
        assert_eq!(c.image_soft_tokens, 560);
        assert_eq!(c.video_soft_tokens, 70);
        assert_eq!(c.video_fps, 2.0);
        assert_eq!(c.video_max_frames, 16);
        assert_eq!(c.hop_length, 160);
        assert!(ProcessorConfig::parse(r#"{"image_processor":{"max_soft_tokens":100}}"#).is_err());
    }
}
