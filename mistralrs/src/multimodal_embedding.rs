use std::{path::PathBuf, sync::Arc};

use anyhow::{ensure, Context, Result};
use embeddinggemma_core::{
    config::ProcessorConfig, DeviceRequest, EmbedError, Embedder, EmbedderOptions, Modalities,
    ModelFiles, ModelSource, Part,
};
use multimodal_tokenizers::Tokenizer;
use tokio::sync::Semaphore;

#[path = "multimodal_embedding_media.rs"]
mod media;

const DEFAULT_MAX_INPUT_TOKENS: usize = 2048;
const MIN_INPUT_TOKENS: usize = 16;
const MAX_BATCH_INPUTS: usize = 32;
const MAX_PARTS_DEPTH: usize = 4;
const MAX_ROW_NODES: usize = 32;
const MAX_BATCH_ENCODED_BYTES: usize = 64 * 1024 * 1024;
const MAX_BATCH_IMAGE_RGB_BYTES: u64 = 128 * 1024 * 1024;

/// A deterministic inference-input failure, downcastable through the returned anyhow error.
/// This Send + Sync marker excludes model compute/device failures; inspect it with anyhow::Error::downcast_ref.
#[derive(Debug, Clone)]
pub struct MultimodalEmbeddingInputError {
    message: String,
}

impl std::fmt::Display for MultimodalEmbeddingInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MultimodalEmbeddingInputError {}

fn mark_input_error(error: anyhow::Error) -> anyhow::Error {
    let message = format!("{error:#}");
    error.context(MultimodalEmbeddingInputError { message })
}

fn classify_core_error(error: EmbedError) -> anyhow::Error {
    let input_failure = match &error {
        EmbedError::InvalidRequest(_) | EmbedError::Tokenizer(_) => true,
        // The pinned core reports a worker panic as Media, not as a deterministic input failure.
        EmbedError::Media(message) => message != "input preprocessing panicked",
        _ => false,
    };
    if input_failure {
        mark_input_error(error.into())
    } else {
        error.into()
    }
}

/// One independent input, producing one vector in the shared embedding space.
#[derive(Debug, Clone)]
pub enum MultimodalEmbeddingInput {
    Text(String),
    /// Encoded image bytes, not raw pixels.
    Image(Vec<u8>),
    /// Encoded audio bytes, not raw samples.
    Audio(Vec<u8>),
    /// Ordered parts flattened recursively into one row; nonempty, at most four Parts levels and 32 tree nodes.
    /// All-text parts concatenate and may truncate; mixed rows reject over-limit text instead of truncating it.
    Parts(Vec<MultimodalEmbeddingInput>),
}

/// Ordered, full-dimensional L2-normalized vectors and per-input truncation flags.
#[derive(Debug, Clone)]
pub struct MultimodalEmbeddingOutput {
    pub vectors: Vec<Vec<f32>>,
    pub truncated: Vec<bool>,
}

/// Loads native EmbeddingGemma 2 safetensors from an explicit local snapshot.
#[derive(Debug, Clone)]
pub struct MultimodalEmbeddingModelBuilder {
    local_snapshot_path: PathBuf,
    force_cpu: bool,
    max_input_tokens: usize,
    modalities: Modalities,
}

impl MultimodalEmbeddingModelBuilder {
    pub fn new(local_snapshot_path: impl Into<PathBuf>) -> Self {
        Self {
            local_snapshot_path: local_snapshot_path.into(),
            force_cpu: false,
            max_input_tokens: DEFAULT_MAX_INPUT_TOKENS,
            modalities: Modalities::ALL,
        }
    }

    pub fn with_force_cpu(mut self, force_cpu: bool) -> Self {
        self.force_cpu = force_cpu;
        self
    }

    /// Includes tokenizer-added special tokens; values below 16 are rejected.
    pub fn with_max_input_tokens(mut self, max_input_tokens: usize) -> Self {
        self.max_input_tokens = max_input_tokens;
        self
    }

    /// Text is always loaded; unavailable checkpoint towers remain unavailable.
    pub fn with_modalities(mut self, vision: bool, audio: bool) -> Self {
        self.modalities = Modalities {
            image: vision,
            audio,
        };
        self
    }

    /// Loading runs on Tokio's blocking pool; started work is not immediately cancellable.
    pub async fn build(self) -> Result<MultimodalEmbeddingModel> {
        tokio::task::spawn_blocking(move || {
            ensure!(
                self.max_input_tokens >= MIN_INPUT_TOKENS,
                "max_input_tokens must be at least {MIN_INPUT_TOKENS}"
            );
            ensure!(
                u32::try_from(self.max_input_tokens).is_ok(),
                "max_input_tokens exceeds the supported range"
            );
            let path = self
                .local_snapshot_path
                .canonicalize()
                .context("cannot resolve local EmbeddingGemma snapshot directory")?;
            ensure!(path.is_dir(), "local snapshot must be a directory");
            let source = ModelSource::Local(path);
            let files = ModelFiles::locate(&source)
                .context("cannot locate native EmbeddingGemma snapshot files")?;
            let processor = match &files.processor_config {
                Some(path) => ProcessorConfig::parse(&std::fs::read_to_string(path)?)?,
                None => ProcessorConfig::default(),
            };
            ensure!(
                processor.sampling_rate == media::AUDIO_SAMPLE_RATE as usize,
                "EmbeddingGemma audio requires a 16 kHz processor"
            );
            ensure!(
                (1..=media::MAX_AUDIO_SAMPLES).contains(&processor.max_audio_samples),
                "processor audio limit must be greater than zero and at most 30 seconds"
            );
            let image_preprocess_rgb_bytes = [
                processor.image_soft_tokens,
                processor.pooling_kernel_size,
                processor.pooling_kernel_size,
                processor.patch_size,
                processor.patch_size,
            ]
            .into_iter()
            .try_fold(3u64, |bytes, factor| bytes.checked_mul(factor as u64))
            .context("processor image preprocessing size overflows")?;
            let mut tokenizer =
                Tokenizer::from_file(&files.tokenizer).map_err(anyhow::Error::msg)?;
            tokenizer
                .with_truncation(None)
                .map_err(anyhow::Error::msg)?;
            tokenizer.with_padding(None);
            let embedder = Embedder::load_files(
                &files,
                &source,
                "embeddinggemma-2".into(),
                EmbedderOptions {
                    device: if self.force_cpu {
                        DeviceRequest::Cpu
                    } else {
                        DeviceRequest::Auto
                    },
                    modalities: Some(self.modalities),
                    max_input_tokens: Some(self.max_input_tokens),
                    ..Default::default()
                },
            )
            .context("cannot load local EmbeddingGemma model")?;
            Ok(MultimodalEmbeddingModel {
                inner: Arc::new(Inner {
                    embedder,
                    tokenizer,
                    max_audio_samples: processor.max_audio_samples,
                    image_preprocess_rgb_bytes,
                }),
                gate: Arc::new(Semaphore::new(1)),
            })
        })
        .await
        .context("EmbeddingGemma load task failed")?
    }
}

#[derive(Debug)]
struct Inner {
    embedder: Embedder,
    tokenizer: Tokenizer,
    max_audio_samples: usize,
    image_preprocess_rgb_bytes: u64,
}

/// Cloneable, Send + Sync model, independent of the mistral.rs runner and scheduler.
#[derive(Debug, Clone)]
pub struct MultimodalEmbeddingModel {
    inner: Arc<Inner>,
    gate: Arc<Semaphore>,
}

impl MultimodalEmbeddingModel {
    /// At most 32 rows and 64 MiB total text/encoded media; mixed rows reject over-limit text without truncation.
    /// No task prefixes are added. Waiting calls can be cancelled; started blocking work may continue.
    pub async fn embed(
        &self,
        inputs: Vec<MultimodalEmbeddingInput>,
    ) -> Result<MultimodalEmbeddingOutput> {
        validate_inputs(&inputs).map_err(mark_input_error)?;
        let permit = self
            .gate
            .clone()
            .acquire_owned()
            .await
            .context("EmbeddingGemma inference queue closed")?;
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            if inputs.is_empty() {
                return Ok(MultimodalEmbeddingOutput {
                    vectors: Vec::new(),
                    truncated: Vec::new(),
                });
            }
            inner
                .validate_image_budget(&inputs)
                .map_err(mark_input_error)?;
            let mut parts = Vec::with_capacity(inputs.len());
            let mut truncated = Vec::with_capacity(inputs.len());
            for (index, input) in inputs.into_iter().enumerate() {
                let (row, was_truncated) = inner
                    .prepare_input(input)
                    .map_err(mark_input_error)
                    .with_context(|| format!("cannot prepare input {index}"))?;
                parts.push(row);
                truncated.push(was_truncated);
            }
            let embeddings = inner
                .embedder
                .embed_inputs(&parts, None, true)
                .map_err(classify_core_error)
                .context("EmbeddingGemma inference failed")?;
            Ok(MultimodalEmbeddingOutput {
                vectors: embeddings.vectors,
                truncated,
            })
        })
        .await
        .context("EmbeddingGemma inference task failed")?
    }
}

fn validate_inputs(inputs: &[MultimodalEmbeddingInput]) -> Result<()> {
    ensure!(
        inputs.len() <= MAX_BATCH_INPUTS,
        "embedding batch exceeds 32 rows"
    );
    let mut encoded_bytes = 0;
    for (index, input) in inputs.iter().enumerate() {
        let mut nodes = 0;
        validate_tree(input, 0, &mut nodes, &mut encoded_bytes)
            .with_context(|| format!("invalid input {index}"))?;
    }
    Ok(())
}

fn validate_tree(
    input: &MultimodalEmbeddingInput,
    depth: usize,
    nodes: &mut usize,
    encoded_bytes: &mut usize,
) -> Result<()> {
    *nodes += 1;
    ensure!(
        *nodes <= MAX_ROW_NODES,
        "input tree exceeds 32 nodes including Parts containers"
    );
    let bytes = match input {
        MultimodalEmbeddingInput::Text(text) => text.len(),
        MultimodalEmbeddingInput::Image(bytes) => {
            ensure!(
                bytes.len() <= media::MAX_IMAGE_BYTES,
                "encoded image exceeds 20 MiB"
            );
            bytes.len()
        }
        MultimodalEmbeddingInput::Audio(bytes) => {
            ensure!(
                bytes.len() <= media::MAX_AUDIO_BYTES,
                "encoded audio exceeds 50 MiB"
            );
            bytes.len()
        }
        MultimodalEmbeddingInput::Parts(parts) => {
            ensure!(!parts.is_empty(), "Parts must not be empty");
            ensure!(depth < MAX_PARTS_DEPTH, "Parts nesting exceeds four levels");
            for part in parts {
                validate_tree(part, depth + 1, nodes, encoded_bytes)?;
            }
            return Ok(());
        }
    };
    ensure!(
        bytes <= MAX_BATCH_ENCODED_BYTES - *encoded_bytes,
        "embedding batch exceeds 64 MiB of text and encoded media"
    );
    *encoded_bytes += bytes;
    Ok(())
}

fn flatten_input(input: MultimodalEmbeddingInput, leaves: &mut Vec<MultimodalEmbeddingInput>) {
    match input {
        MultimodalEmbeddingInput::Parts(parts) => {
            for part in parts {
                flatten_input(part, leaves);
            }
        }
        leaf => leaves.push(leaf),
    }
}

impl Inner {
    fn validate_image_budget(&self, inputs: &[MultimodalEmbeddingInput]) -> Result<()> {
        let mut rgb_bytes = 0;
        for (index, input) in inputs.iter().enumerate() {
            self.inspect_image_tree(input, &mut rgb_bytes)
                .with_context(|| format!("invalid image budget/header in input {index}"))?;
        }
        Ok(())
    }

    fn inspect_image_tree(
        &self,
        input: &MultimodalEmbeddingInput,
        rgb_bytes: &mut u64,
    ) -> Result<()> {
        match input {
            MultimodalEmbeddingInput::Image(bytes) => {
                let decoded_rgb_bytes = media::image_rgb_bytes(bytes)?;
                // Reserving resized pixels too bounds patchification even for a batch of tiny images.
                let charge = decoded_rgb_bytes.max(self.image_preprocess_rgb_bytes);
                ensure!(
                    charge <= MAX_BATCH_IMAGE_RGB_BYTES - *rgb_bytes,
                    "batch image RGB/preprocessing budget exceeds 128 MiB"
                );
                *rgb_bytes += charge;
            }
            MultimodalEmbeddingInput::Parts(parts) => {
                for part in parts {
                    self.inspect_image_tree(part, rgb_bytes)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn prepare_input(&self, input: MultimodalEmbeddingInput) -> Result<(Vec<Part>, bool)> {
        let mut leaves = Vec::new();
        flatten_input(input, &mut leaves);
        let info = self.embedder.info();
        if leaves
            .iter()
            .all(|part| matches!(part, MultimodalEmbeddingInput::Text(_)))
        {
            let text: String = leaves
                .into_iter()
                .map(|part| match part {
                    MultimodalEmbeddingInput::Text(text) => text,
                    _ => unreachable!("all leaves are text"),
                })
                .collect();
            let raw = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(anyhow::Error::msg)?;
            return Ok((
                vec![Part::Text(text)],
                raw.len() > info.max_input_tokens as usize,
            ));
        }
        let mut text_tokens = 0;
        for part in &leaves {
            if let MultimodalEmbeddingInput::Text(text) = part {
                text_tokens += self
                    .tokenizer
                    .encode(text.as_str(), false)
                    .map_err(anyhow::Error::msg)?
                    .len();
                // Upstream encodes each mixed-row fragment with a truncation-enabled tokenizer.
                ensure!(
                    text_tokens <= info.max_input_tokens as usize,
                    "mixed input text exceeds the token limit; mixed rows cannot be truncated"
                );
            }
        }
        let mut row = Vec::with_capacity(leaves.len());
        let mut truncated = false;
        for part in leaves {
            row.push(match part {
                MultimodalEmbeddingInput::Text(text) => Part::Text(text),
                MultimodalEmbeddingInput::Image(bytes) => {
                    ensure!(info.modalities.image, "vision tower is not loaded");
                    Part::Image(media::decode_image(&bytes).context("cannot decode image part")?)
                }
                MultimodalEmbeddingInput::Audio(bytes) => {
                    ensure!(info.modalities.audio, "audio tower is not loaded");
                    let (samples, clipped) = media::decode_audio(bytes, self.max_audio_samples)
                        .context("cannot decode audio part")?;
                    truncated |= clipped;
                    Part::Audio(samples)
                }
                MultimodalEmbeddingInput::Parts(_) => {
                    unreachable!("Parts containers were flattened")
                }
            });
        }
        Ok((row, truncated))
    }
}
