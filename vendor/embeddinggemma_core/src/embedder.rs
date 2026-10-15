//! [`Embedder`]: tokenizer + processors + model, with length-sorted
//! micro-batching, mean pooling, Matryoshka truncation and normalization.

use crate::backend::Backend;
use crate::backend::candle::{self as cb, CandleBackend};
use crate::config::{
    ModelConfig, ProcessorConfig, SUPPORTED_SOFT_TOKENS, sentence_transformers_prompts,
};
use crate::content::Content;
use crate::device::{DeviceRequest, MLX_AVAILABLE, Precision};
use crate::error::{EmbedError, Result};
use crate::loader::{ModelFiles, ModelSource, model_dir, resolve_source, source_id};
use crate::media::{self, MediaKind, ToolOptions, audio, image as img, video};
use crate::model::{EmbeddingGemma2, Engine, MediaInput, PreparedRow};
use crate::registry::{self, Modalities, PROVIDER};
use crate::task::{
    ContentEmbeddingsRequest, EmbeddingsRequest, EmbeddingsResponse, PromptChoice, Prompts,
    TokenCounts, Usage, finish_vector, resolve_dimensions,
};
use image::RgbImage;
use serde::Serialize;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

/// How to load and drive a model.
#[derive(Debug, Clone)]
pub struct EmbedderOptions {
    pub device: DeviceRequest,
    /// Activation / weight precision. `None` → bf16 on accelerators (CUDA,
    /// Metal, MLX), f32 on CPU. The model card warns against f16.
    pub dtype: Option<Precision>,
    /// Towers to load. `None` → what the model name implies (`270m` = text
    /// only, …), else all.
    pub modalities: Option<Modalities>,
    /// Text inputs longer than this are truncated (multimodal inputs that
    /// do not fit are rejected). `None` → 8 192.
    pub max_input_tokens: Option<usize>,
    /// Most rows in one forward pass.
    pub max_batch_size: usize,
    /// Most padded tokens (`rows × longest row`) in one forward pass.
    pub max_batch_tokens: usize,
    /// Soft tokens per image (70, 140, 280, 560, 1120). `None` → the
    /// checkpoint's (280).
    pub image_soft_tokens: Option<usize>,
    /// Soft tokens per video frame. `None` → the checkpoint's (140).
    pub video_soft_tokens: Option<usize>,
    /// Longest audio clip, in seconds. `None` → the feature extractor's 30 s.
    pub max_audio_seconds: Option<f64>,
    /// `ffmpeg` / `pdftoppm` and video / PDF sampling.
    pub tools: ToolOptions,
}

impl Default for EmbedderOptions {
    fn default() -> Self {
        Self {
            device: DeviceRequest::Auto,
            dtype: None,
            modalities: None,
            max_input_tokens: None,
            max_batch_size: 32,
            max_batch_tokens: 16_384,
            image_soft_tokens: None,
            video_soft_tokens: None,
            max_audio_seconds: None,
            tools: ToolOptions::default(),
        }
    }
}

/// What is loaded, for logs and `GET /v1/models`.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    /// Bare id (`embeddinggemma-2`).
    pub id: String,
    /// `embeddinggemma::embeddinggemma-2`.
    pub qualified_id: String,
    pub display_name: String,
    /// Hub repo or local path the weights came from.
    pub source: String,
    /// Parameters of the loaded towers (`740M`, `270M`, …).
    pub parameters: String,
    pub modalities: Modalities,
    pub embedding_dimensions: u32,
    pub matryoshka_dimensions: Vec<u32>,
    pub max_input_tokens: u32,
    /// `candle` or `mlx`.
    pub backend: &'static str,
    /// `cpu`, `cuda`, `metal` or `mlx`.
    pub device: String,
    pub dtype: Precision,
    pub image_soft_tokens: usize,
    pub video_soft_tokens: usize,
    pub max_audio_seconds: f64,
    pub load_seconds: f64,
}

/// Vectors plus the tokens that produced them.
#[derive(Debug, Clone, PartialEq)]
pub struct Embeddings {
    pub vectors: Vec<Vec<f32>>,
    pub tokens: TokenCounts,
}

/// One decoded input part.
#[derive(Debug, Clone)]
pub enum Part {
    Text(String),
    Image(RgbImage),
    /// 16 kHz mono samples.
    Audio(Vec<f32>),
    /// Frames in order (already sampled).
    Video(Vec<RgbImage>),
}

/// Special token ids the input builder needs.
#[derive(Debug, Clone, Copy)]
struct Ids {
    bos: u32,
    eos: u32,
    image: u32,
    video: u32,
    audio: u32,
    boi: u32,
    eoi: u32,
    boa: u32,
    eoa: u32,
}

/// A loaded EmbeddingGemma 2. `Send + Sync`; forward passes are serialized
/// through an internal lock so one instance can be shared behind an `Arc`
/// (media decoding and tokenization run outside the lock).
pub struct Embedder {
    engine: Box<dyn Engine>,
    tokenizer: Tokenizer,
    prompts: Prompts,
    processor: ProcessorConfig,
    ids: Ids,
    space_ids: Vec<u32>,
    info: ModelInfo,
    options: EmbedderOptions,
    compute: Mutex<()>,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder")
            .field("info", &self.info)
            .finish()
    }
}

/// The backend a device request resolves to.
enum Target {
    Candle(candle_core::Device),
    #[allow(dead_code)]
    Mlx,
}

fn resolve_target(request: DeviceRequest) -> Result<Target> {
    match request {
        DeviceRequest::Mlx if MLX_AVAILABLE => Ok(Target::Mlx),
        DeviceRequest::Mlx => Err(EmbedError::Device(
            "mlx requested but this binary was built without the `mlx` feature (macOS only)".into(),
        )),
        DeviceRequest::Auto | DeviceRequest::Gpu => {
            if cfg!(feature = "cuda")
                && let Ok(d) = cb::select_device(DeviceRequest::Cuda(0))
            {
                return Ok(Target::Candle(d));
            }
            if MLX_AVAILABLE {
                return Ok(Target::Mlx);
            }
            Ok(Target::Candle(cb::select_device(request)?))
        }
        other => Ok(Target::Candle(cb::select_device(other)?)),
    }
}

fn build_engine<B: Backend>(
    be: B,
    cfg: ModelConfig,
    files: &ModelFiles,
    modalities: Modalities,
    max_positions: usize,
) -> Result<Box<dyn Engine>> {
    Ok(Box::new(EmbeddingGemma2::load(
        be,
        cfg,
        &files.weights,
        modalities,
        max_positions,
    )?))
}

impl Embedder {
    /// Load by registry name (`embeddinggemma-2`, `270m`, `440m`, `570m`,
    /// `740m`), Hub repo (`google/embeddinggemma-2[@rev]`) or local
    /// directory. A copy under the model directory
    /// (`$EMBEDDINGGEMMA_MODEL_DIR`, default `./models`) is used before
    /// anything is downloaded.
    pub fn load(name: &str, options: EmbedderOptions) -> Result<Self> {
        Self::load_from(name, &model_dir(), options)
    }

    /// [`Embedder::load`] with an explicit model directory.
    pub fn load_from(name: &str, model_dir: &Path, options: EmbedderOptions) -> Result<Self> {
        let resolved = registry::resolve(name);
        let source = resolve_source(name, model_dir);
        let files = ModelFiles::locate(&source)?;
        let id = resolved
            .map(|(s, _)| s.id.to_string())
            .unwrap_or_else(|| source_id(&source));
        let mut options = options;
        if options.modalities.is_none() {
            options.modalities = resolved.and_then(|(_, m)| m);
        }
        Self::load_files(&files, &source, id, options)
    }

    /// Load from already-located files.
    pub fn load_files(
        files: &ModelFiles,
        source: &ModelSource,
        id: String,
        options: EmbedderOptions,
    ) -> Result<Self> {
        let started = Instant::now();
        let cfg = ModelConfig::parse(&std::fs::read_to_string(&files.config)?)?;
        let mut processor = match &files.processor_config {
            Some(p) => ProcessorConfig::parse(&std::fs::read_to_string(p)?)?,
            None => ProcessorConfig::default(),
        };
        if let Some(n) = options.image_soft_tokens {
            if !SUPPORTED_SOFT_TOKENS.contains(&n) {
                return Err(EmbedError::InvalidRequest(format!(
                    "image soft tokens must be one of {SUPPORTED_SOFT_TOKENS:?}, got {n}"
                )));
            }
            processor.image_soft_tokens = n;
        }
        if let Some(n) = options.video_soft_tokens {
            if !SUPPORTED_SOFT_TOKENS.contains(&n) {
                return Err(EmbedError::InvalidRequest(format!(
                    "video soft tokens must be one of {SUPPORTED_SOFT_TOKENS:?}, got {n}"
                )));
            }
            processor.video_soft_tokens = n;
        }
        if let Some(s) = options.max_audio_seconds {
            processor.max_audio_samples = (s.max(0.1) * processor.sampling_rate as f64) as usize;
        }
        let prompts = match &files.sentence_transformers_config {
            Some(p) => Prompts::with_overrides(sentence_transformers_prompts(
                &std::fs::read_to_string(p)?,
            )?),
            None => Prompts::default(),
        };
        let spec = registry::resolve(&id).map(|(s, _)| s);
        let documented = spec.map(|s| s.max_input_tokens as usize).unwrap_or(8192);
        let max_input_tokens = options.max_input_tokens.unwrap_or(documented).max(16);
        let modalities = {
            let mut m = options.modalities.unwrap_or(Modalities::ALL);
            m.image &= cfg.vision_config.is_some();
            m.audio &= cfg.audio_config.is_some();
            m
        };

        let mut tokenizer = Tokenizer::from_file(&files.tokenizer)?;
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: max_input_tokens,
            strategy: TruncationStrategy::LongestFirst,
            direction: TruncationDirection::Right,
            stride: 0,
        }))?;
        tokenizer.with_padding(None);
        let space_ids = tokenizer.encode(" ", false)?.get_ids().to_vec();
        let tc = &cfg.text_config;
        let ids = Ids {
            bos: tc.bos_token_id.unwrap_or(2),
            eos: tc.eos_token_id.unwrap_or(1),
            image: cfg.image_token_id,
            video: cfg.video_token_id,
            audio: cfg.audio_token_id,
            boi: cfg.boi_token_id,
            eoi: cfg.eoi_token_id,
            boa: cfg.boa_token_id,
            eoa: cfg.eoa_token_id,
        };
        let embedding_dim = tc.embedding_dim as u32;

        let (engine, backend): (Box<dyn Engine>, &'static str) =
            match resolve_target(options.device)? {
                Target::Candle(device) => {
                    let precision = options.dtype.unwrap_or(if device.is_cpu() {
                        Precision::F32
                    } else {
                        Precision::BF16
                    });
                    let be = CandleBackend::new(device, precision);
                    (
                        build_engine(be, cfg, files, modalities, max_input_tokens)?,
                        "candle",
                    )
                }
                Target::Mlx => {
                    #[cfg(all(feature = "mlx", target_os = "macos"))]
                    {
                        let be = crate::backend::mlx::MlxBackend::new(
                            options.dtype.unwrap_or(Precision::BF16),
                        )?;
                        (
                            build_engine(be, cfg, files, modalities, max_input_tokens)?,
                            "mlx",
                        )
                    }
                    #[cfg(not(all(feature = "mlx", target_os = "macos")))]
                    unreachable!("resolve_target only returns Mlx when MLX is compiled in")
                }
            };

        let info = ModelInfo {
            qualified_id: format!("{PROVIDER}::{id}"),
            display_name: spec
                .map(|s| s.display_name.to_string())
                .unwrap_or_else(|| id.clone()),
            id,
            source: source.describe(),
            parameters: modalities.parameters().to_string(),
            modalities: engine.modalities(),
            embedding_dimensions: embedding_dim,
            matryoshka_dimensions: spec
                .map(|s| s.matryoshka_dimensions.to_vec())
                .unwrap_or_else(|| vec![embedding_dim]),
            max_input_tokens: max_input_tokens as u32,
            backend,
            device: engine.device_name(),
            dtype: engine.precision(),
            image_soft_tokens: processor.image_soft_tokens,
            video_soft_tokens: processor.video_soft_tokens,
            max_audio_seconds: processor.max_audio_samples as f64 / processor.sampling_rate as f64,
            load_seconds: started.elapsed().as_secs_f64(),
        };
        Ok(Self {
            engine,
            tokenizer,
            prompts,
            processor,
            ids,
            space_ids,
            info,
            options,
            compute: Mutex::new(()),
        })
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    pub fn options(&self) -> &EmbedderOptions {
        &self.options
    }

    pub fn prompts(&self) -> &Prompts {
        &self.prompts
    }

    /// One forward pass per loaded tower so kernels / allocators are ready
    /// before the first real request.
    pub fn warmup(&self) -> Result<()> {
        let mut inputs = vec![vec![Part::Text("warmup".into())]];
        if self.info.modalities.image {
            inputs.push(vec![Part::Image(RgbImage::new(48, 48))]);
        }
        if self.info.modalities.audio {
            inputs.push(vec![Part::Audio(vec![0.0; 16_000])]);
        }
        self.embed_inputs(&inputs, None, true).map(|_| ())
    }

    /// Token ids for one text (with `<bos>` / `<eos>`, after truncation).
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(text, true)?.get_ids().to_vec())
    }

    /// `POST /v1/embeddings`: prefix every text with the task's prompt, embed,
    /// truncate to `dimensions`, normalize.
    pub fn embed_request(&self, request: &EmbeddingsRequest) -> Result<EmbeddingsResponse> {
        if request.input.is_empty() {
            return Err(EmbedError::InvalidRequest(
                "`input` must contain at least one text".into(),
            ));
        }
        let prefix = request.prompt_choice().prefix(&self.prompts);
        let texts: Vec<String> = request
            .input
            .as_slice()
            .iter()
            .map(|t| format!("{prefix}{t}"))
            .collect();
        let e = self.embed_texts_with(
            &texts,
            request.dimensions,
            request.normalize.unwrap_or(true),
        )?;
        Ok(EmbeddingsResponse {
            external_id: None,
            output: e.vectors,
            usage: Some(Usage::from_counts(&e.tokens)),
        })
    }

    /// `POST /v1/embeddings/contents`: one vector per item; text-only items
    /// get the task prompt, multimodal items none (unless `prompt` is set).
    pub fn embed_contents(&self, request: &ContentEmbeddingsRequest) -> Result<EmbeddingsResponse> {
        if request.input.is_empty() {
            return Err(EmbedError::InvalidRequest(
                "`input` must contain at least one item".into(),
            ));
        }
        let choice = request.prompt_choice();
        let inputs = par_map(request.input.as_slice(), |item| {
            self.content_parts(item, &choice)
        })?;
        let e = self.embed_inputs(
            &inputs,
            request.dimensions,
            request.normalize.unwrap_or(true),
        )?;
        Ok(EmbeddingsResponse {
            external_id: None,
            output: e.vectors,
            usage: Some(Usage::from_counts(&e.tokens)),
        })
    }

    /// Embed texts exactly as given (no prompt), full size.
    pub fn embed_texts(&self, texts: &[String], normalize: bool) -> Result<Embeddings> {
        self.embed_texts_with(texts, None, normalize)
    }

    fn embed_texts_with(
        &self,
        texts: &[String],
        dims: Option<u32>,
        normalize: bool,
    ) -> Result<Embeddings> {
        if texts.is_empty() {
            return Ok(Embeddings {
                vectors: Vec::new(),
                tokens: TokenCounts::default(),
            });
        }
        let encodings = self.tokenizer.encode_batch(texts.to_vec(), true)?;
        let rows: Vec<PreparedRow> = encodings
            .iter()
            .map(|e| PreparedRow {
                ids: e.get_ids().to_vec(),
                media: Vec::new(),
            })
            .collect();
        let tokens = TokenCounts {
            text: rows.iter().map(|r| r.ids.len()).sum(),
            ..Default::default()
        };
        self.run(rows, tokens, dims, normalize)
    }

    /// Embed decoded inputs: one vector per input, each input's parts in
    /// order. All-text inputs are embedded exactly as given (no prompt).
    pub fn embed_inputs(
        &self,
        inputs: &[Vec<Part>],
        dims: Option<u32>,
        normalize: bool,
    ) -> Result<Embeddings> {
        let prepared = par_map(inputs, |parts| self.prepare_row(parts))?;
        let mut tokens = TokenCounts::default();
        let rows = prepared
            .into_iter()
            .map(|(row, counts)| {
                tokens.add(&counts);
                row
            })
            .collect();
        self.run(rows, tokens, dims, normalize)
    }

    fn run(
        &self,
        rows: Vec<PreparedRow>,
        tokens: TokenCounts,
        dims: Option<u32>,
        normalize: bool,
    ) -> Result<Embeddings> {
        let lengths: Vec<usize> = rows.iter().map(|r| r.ids.len()).collect();
        let batches = plan_batches(
            &lengths,
            self.options.max_batch_size,
            self.options.max_batch_tokens,
        );
        let pooled = {
            let _guard = self.compute.lock().unwrap_or_else(|p| p.into_inner());
            self.engine.embed_rows(&rows, &batches)?
        };
        let dims = resolve_dimensions(
            dims,
            &self.info.matryoshka_dimensions,
            self.info.embedding_dimensions,
        ) as usize;
        Ok(Embeddings {
            vectors: pooled
                .into_iter()
                .map(|v| finish_vector(v, dims, normalize))
                .collect(),
            tokens,
        })
    }

    /// Token ids + media for one input, mirroring `EmbeddingGemma2Processor`:
    /// `<bos>`, then the parts in order — text verbatim, each image as
    /// `<|image>` + soft-token placeholders + `<image|>`, each video frame
    /// likewise with `<|video|>` placeholders, audio between `<|audio>` and
    /// `<audio|>` — then `<eos>`. Media-only inputs separate their parts
    /// with a space (the processor's `" ".join`), inputs with text do not
    /// (the chat template's concatenation).
    pub fn prepare_row(&self, parts: &[Part]) -> Result<(PreparedRow, TokenCounts)> {
        if parts.is_empty() {
            return Err(EmbedError::InvalidRequest("an input has no parts".into()));
        }
        let mut counts = TokenCounts::default();
        if parts.iter().all(|p| matches!(p, Part::Text(_))) {
            let text: String = parts
                .iter()
                .map(|p| match p {
                    Part::Text(t) => t.as_str(),
                    _ => unreachable!(),
                })
                .collect();
            let ids = self.tokenize(&text)?;
            counts.text = ids.len();
            return Ok((
                PreparedRow {
                    ids,
                    media: Vec::new(),
                },
                counts,
            ));
        }
        let m = self.info.modalities;
        let ids = self.ids;
        let any_text = parts.iter().any(|p| matches!(p, Part::Text(_)));
        let pc = &self.processor;
        let mut row = PreparedRow {
            ids: vec![ids.bos],
            media: Vec::new(),
        };
        for (i, part) in parts.iter().enumerate() {
            if i > 0 && !any_text {
                row.ids.extend_from_slice(&self.space_ids);
            }
            match part {
                Part::Text(t) => row
                    .ids
                    .extend_from_slice(self.tokenizer.encode(t.as_str(), false)?.get_ids()),
                Part::Image(image) => {
                    require(m.image, "image")?;
                    let p = img::preprocess_image(
                        image,
                        pc.image_soft_tokens,
                        pc.patch_size,
                        pc.pooling_kernel_size,
                    )?;
                    let n = p.soft_tokens();
                    push_block(&mut row.ids, ids.boi, ids.image, n, ids.eoi);
                    row.media.push(MediaInput::Image(p));
                    counts.image += n;
                }
                Part::Video(frames) => {
                    require(m.image, "video")?;
                    if frames.is_empty() {
                        return Err(EmbedError::InvalidRequest("a video has no frames".into()));
                    }
                    for frame in frames {
                        let p = img::preprocess_image(
                            frame,
                            pc.video_soft_tokens,
                            pc.patch_size,
                            pc.pooling_kernel_size,
                        )?;
                        let n = p.soft_tokens();
                        push_block(&mut row.ids, ids.boi, ids.video, n, ids.eoi);
                        row.media.push(MediaInput::Image(p));
                        counts.video += n;
                    }
                }
                Part::Audio(samples) => {
                    require(m.audio, "audio")?;
                    let f = audio::log_mel(samples, pc)?;
                    let n = f.soft_tokens();
                    push_block(&mut row.ids, ids.boa, ids.audio, n, ids.eoa);
                    row.media.push(MediaInput::Audio(f));
                    counts.audio += n;
                }
            }
        }
        row.ids.push(ids.eos);
        counts.text = row.ids.len() - counts.image - counts.video - counts.audio;
        if row.ids.len() > self.info.max_input_tokens as usize {
            return Err(EmbedError::InvalidRequest(format!(
                "input is {} tokens ({} image, {} video, {} audio); the limit is {} — use fewer \
                 or smaller media (image soft tokens: {}, video: {} frames × {} tokens)",
                row.ids.len(),
                counts.image,
                counts.video,
                counts.audio,
                self.info.max_input_tokens,
                pc.image_soft_tokens,
                self.options.tools.video_max_frames,
                pc.video_soft_tokens
            )));
        }
        Ok((row, counts))
    }

    /// Decode one content item into parts. Items that end up all text get
    /// the task prefix; others get only an explicit `prompt`.
    pub fn content_parts(&self, item: &Content, choice: &PromptChoice) -> Result<Vec<Part>> {
        let mut parts = Vec::new();
        self.collect_parts(item, &mut parts)?;
        if parts.iter().all(|p| matches!(p, Part::Text(_))) {
            let prefix = choice.prefix(&self.prompts);
            let mut text = prefix;
            for p in parts {
                if let Part::Text(t) = p {
                    text.push_str(&t);
                }
            }
            return Ok(vec![Part::Text(text)]);
        }
        if let Some(p) = choice.prompt.as_ref().filter(|p| !p.is_empty()) {
            parts.insert(0, Part::Text(p.clone()));
        }
        Ok(parts)
    }

    fn collect_parts(&self, item: &Content, out: &mut Vec<Part>) -> Result<()> {
        let tools = &self.options.tools;
        match item {
            Content::Text { text } => out.push(Part::Text(text.clone())),
            Content::Image { data, format } => {
                out.push(Part::Image(img::decode_image(&data.data)?));
                let _ = format;
            }
            Content::Audio { data, format } => {
                let hint = format.as_deref().or(data.mime.as_deref());
                out.push(Part::Audio(self.decode_audio(&data.data, hint)?));
            }
            Content::Video {
                data,
                format,
                frames,
            } => {
                let frames = if !frames.is_empty() {
                    let decoded = frames
                        .iter()
                        .map(|f| img::decode_image(&f.data))
                        .collect::<Result<Vec<_>>>()?;
                    video::cap_frames(decoded, tools.video_max_frames)
                } else if let Some(d) = data {
                    let hint = format.as_deref().or(d.mime.as_deref());
                    video::decode_video(&d.data, media::extension_hint(hint).as_deref(), tools)?
                } else {
                    return Err(EmbedError::InvalidRequest(
                        "a video item needs `data` or `frames`".into(),
                    ));
                };
                out.push(Part::Video(frames));
            }
            Content::File { data, name } => {
                let hint = name.as_deref().or(data.mime.as_deref());
                match media::detect(&data.data, hint) {
                    Some(MediaKind::Image) => out.push(Part::Image(img::decode_image(&data.data)?)),
                    Some(MediaKind::Audio) => {
                        out.push(Part::Audio(self.decode_audio(&data.data, hint)?))
                    }
                    Some(MediaKind::Video) => out.push(Part::Video(video::decode_video(
                        &data.data,
                        media::extension_hint(hint).as_deref(),
                        tools,
                    )?)),
                    Some(MediaKind::Pdf) => {
                        for page in video::render_pdf(&data.data, tools)? {
                            out.push(Part::Image(page));
                        }
                    }
                    Some(MediaKind::Text) => out
                        .push(Part::Text(String::from_utf8(data.data.clone()).map_err(
                            |_| EmbedError::Media("text file is not UTF-8".into()),
                        )?)),
                    None => {
                        return Err(EmbedError::Media(format!(
                            "unrecognised file type{} ({} bytes)",
                            name.as_deref()
                                .map(|n| format!(" `{n}`"))
                                .unwrap_or_default(),
                            data.data.len()
                        )));
                    }
                }
            }
            Content::Parts { parts } => {
                for p in parts {
                    self.collect_parts(p, out)?;
                }
            }
        }
        Ok(())
    }

    /// Decode, mix to mono, cut to the clip limit, resample to 16 kHz.
    pub fn decode_audio(&self, bytes: &[u8], hint: Option<&str>) -> Result<Vec<f32>> {
        let (samples, rate) = audio::decode_audio(bytes, media::extension_hint(hint).as_deref())?;
        let target = self.processor.sampling_rate as u32;
        // Resample a little more than the limit; log_mel truncates exactly.
        let keep = (self.processor.max_audio_samples as u64 * rate as u64 / target as u64) as usize
            + rate as usize;
        let samples = &samples[..samples.len().min(keep)];
        Ok(audio::resample(samples, rate, target))
    }
}

/// `items.iter().map(f).collect()` on scoped threads (one contiguous chunk
/// per core): media decoding and resizing are CPU-bound and independent per
/// item. Results keep the input order; the first error wins.
fn par_map<T: Sync, U: Send>(items: &[T], f: impl Fn(&T) -> Result<U> + Sync) -> Result<Vec<U>> {
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(items.len());
    if workers <= 1 {
        return items.iter().map(&f).collect();
    }
    let chunk = items.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|c| scope.spawn(|| c.iter().map(&f).collect::<Result<Vec<U>>>()))
            .collect();
        let mut out = Vec::with_capacity(items.len());
        for h in handles {
            out.extend(
                h.join()
                    .map_err(|_| EmbedError::Media("input preprocessing panicked".into()))??,
            );
        }
        Ok(out)
    })
}

fn require(loaded: bool, what: &str) -> Result<()> {
    if loaded {
        Ok(())
    } else {
        Err(EmbedError::InvalidRequest(format!(
            "{what} input needs the {} tower, which this model was loaded without",
            if what == "audio" { "audio" } else { "vision" }
        )))
    }
}

fn push_block(ids: &mut Vec<u32>, begin: u32, placeholder: u32, n: usize, end: u32) {
    ids.push(begin);
    ids.extend(std::iter::repeat_n(placeholder, n));
    ids.push(end);
}

/// Group row indices into forward passes: longest first, so padding waste is
/// minimal and the first batch is the worst case for memory. A row joins a
/// batch only while the batch stays under `max_batch_size` rows and
/// `max_batch_tokens` padded tokens **and** the row is at least half as long
/// as the batch's longest row.
pub fn plan_batches(
    lengths: &[usize],
    max_batch_size: usize,
    max_batch_tokens: usize,
) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by(|&a, &b| lengths[b].cmp(&lengths[a]).then(a.cmp(&b)));
    let max_batch_size = max_batch_size.max(1);
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut longest = 0usize;
    for idx in order {
        let len = lengths[idx].max(1);
        let fits = current.is_empty()
            || (current.len() < max_batch_size
                && (current.len() + 1) * longest <= max_batch_tokens
                && len * 2 >= longest);
        if !fits {
            batches.push(std::mem::take(&mut current));
            longest = 0;
        }
        longest = longest.max(len);
        current.push(idx);
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_by_size_tokens_and_length() {
        assert_eq!(
            plan_batches(&[10, 6, 8, 5, 9], 2, 1000),
            vec![vec![0, 4], vec![2, 1], vec![3]]
        );
        assert_eq!(plan_batches(&[500, 400], 32, 100), vec![vec![0], vec![1]]);
        assert_eq!(
            plan_batches(&[12, 3000, 14, 11], 32, 16_384),
            vec![vec![1], vec![2, 0, 3]]
        );
        assert!(plan_batches(&[], 32, 100).is_empty());
    }
}
