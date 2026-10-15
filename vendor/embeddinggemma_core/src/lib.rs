//! # embeddinggemma_core
//!
//! Google's open-weights multimodal embedding model
//! [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) in
//! Rust: text, images, video, PDF pages and audio into one 768-d space
//! (Matryoshka 512 / 256 / 128), on
//!
//! - **candle** — CPU, CUDA (`--features cuda`) and Metal (`--features metal`);
//! - **Apple MLX** — `--features mlx` on macOS / Apple Silicon, the Metal 4
//!   stack that drives the M5 GPU Neural Accelerators.
//!
//! The model (Gemma 4 text encoder, vision and audio towers) is written once
//! against the small [`backend::Backend`] trait; the request / response types
//! and task vocabulary are wire-compatible with GAISe's embedding contract.
//!
//! ```no_run
//! use embeddinggemma_core::{Content, ContentEmbeddingsRequest, Embedder, EmbedderOptions, EmbeddingTask, EmbeddingsRequest};
//!
//! let embedder = Embedder::load("embeddinggemma-2", EmbedderOptions::default())?;
//! let query = embedder.embed_request(&EmbeddingsRequest {
//!     input: vec!["a red bicycle".to_string()].into(),
//!     task: Some(EmbeddingTask::Query),
//!     ..Default::default()
//! })?;
//! let photo = std::fs::read("bike.jpg")?;
//! let image = embedder.embed_contents(&ContentEmbeddingsRequest {
//!     input: vec![Content::Image { data: photo.into(), format: None }].into(),
//!     ..Default::default()
//! })?;
//! let score: f32 = query.output[0].iter().zip(&image.output[0]).map(|(a, b)| a * b).sum();
//! # let _ = score;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod backend;
pub mod config;
pub mod content;
pub mod device;
pub mod embedder;
pub mod error;
pub mod loader;
pub mod media;
pub mod model;
pub mod registry;
pub mod task;

pub use content::{Bytes, Content};
pub use device::{DeviceRequest, MLX_AVAILABLE, Precision, compiled_backends, parse_precision};
pub use embedder::{Embedder, EmbedderOptions, Embeddings, ModelInfo, Part, plan_batches};
pub use error::{EmbedError, Result};
pub use loader::{
    MODEL_DIR_ENV, ModelFiles, ModelSource, export_to_dir, model_dir, resolve_source,
};
pub use media::ToolOptions;
pub use registry::{DEFAULT_MODEL_ID, MODELS, Modalities, ModelSpec, PROVIDER};
pub use task::{
    ContentEmbeddingsRequest, EmbeddingTask, EmbeddingsRequest, EmbeddingsResponse, OneOrMany,
    PromptChoice, Prompts, TokenCounts, Usage,
};
