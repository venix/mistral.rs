//! EmbeddingGemma 2, written once against [`Backend`]:
//!
//! - [`text::TextModel`] — the Gemma 4 text encoder (bidirectional, 5:1
//!   sliding / full layers, per-layer embeddings) with its 512 → 768
//!   `embedding_projection`;
//! - [`vision::VisionTower`] — the Gemma 4 vision encoder (16×16 patches,
//!   axial 2-D RoPE, 3×3 pooling) + `embed_vision`;
//! - [`audio::AudioTower`] — the Gemma 4 USM-style Conformer (mel → 2×
//!   stride-2 conv → 12 Conformer layers with chunked local attention) +
//!   `embed_audio`.
//!
//! [`EmbeddingGemma2`] merges soft tokens from the towers into the text
//! sequence at the placeholder positions and mean-pools the result, exactly
//! like `EmbeddingGemma2Model` + sentence-transformers' `Pooling(mean)`.

pub mod audio;
pub mod text;
pub mod vision;

use crate::backend::Backend;
use crate::config::ModelConfig;
use crate::device::Precision;
use crate::error::{EmbedError, Result};
use crate::media::{AudioFeatures, ImagePatches};
use crate::registry::Modalities;
use std::path::PathBuf;

/// A bias-optional linear layer, `w: [out, in]`.
#[derive(Debug, Clone)]
pub struct Linear<T> {
    pub w: T,
    pub b: Option<T>,
}

impl<T: Clone> Linear<T> {
    pub fn forward<B: Backend<Tensor = T>>(&self, be: &B, x: &T) -> Result<T> {
        be.linear(x, &self.w, self.b.as_ref())
    }
}

/// Gemma 4's `ClippableLinear`: clamp the input, project, clamp the output.
/// Bounds are `None` when the checkpoint has none (or they are infinite).
#[derive(Debug, Clone)]
pub struct ClippedLinear<T> {
    pub linear: Linear<T>,
    pub input: Option<(f64, f64)>,
    pub output: Option<(f64, f64)>,
}

impl<T: Clone> ClippedLinear<T> {
    pub fn forward<B: Backend<Tensor = T>>(&self, be: &B, x: &T) -> Result<T> {
        let x = match self.input {
            Some((lo, hi)) => be.clamp(x, lo, hi)?,
            None => x.clone(),
        };
        let y = self.linear.forward(be, &x)?;
        match self.output {
            Some((lo, hi)) => be.clamp(&y, lo, hi),
            None => Ok(y),
        }
    }
}

/// Typed access to an opened checkpoint.
pub struct Loader<'a, B: Backend> {
    pub be: &'a B,
    pub weights: &'a B::Weights,
}

impl<B: Backend> Loader<'_, B> {
    pub fn has(&self, name: &str) -> bool {
        self.be.has_weight(self.weights, name)
    }

    /// A parameter in the compute precision, shape-checked.
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<B::Tensor> {
        self.checked(name, shape, false)
    }

    /// A parameter in f32 (norm weights), shape-checked.
    pub fn get_f32(&self, name: &str, shape: &[usize]) -> Result<B::Tensor> {
        self.checked(name, shape, true)
    }

    fn checked(&self, name: &str, shape: &[usize], f32: bool) -> Result<B::Tensor> {
        let t = self.be.weight(self.weights, name, f32)?;
        let got = self.be.shape(&t);
        if got != shape {
            return Err(EmbedError::Config(format!(
                "checkpoint tensor `{name}` is {got:?}, expected {shape:?}"
            )));
        }
        Ok(t)
    }

    /// A scalar buffer as a host f64.
    pub fn scalar(&self, name: &str) -> Result<f64> {
        let t = self.be.weight(self.weights, name, true)?;
        let v = self.be.to_f32_vec(&t)?;
        match v.as_slice() {
            [x] => Ok(*x as f64),
            _ => Err(EmbedError::Config(format!(
                "checkpoint tensor `{name}` has {} values, expected a scalar",
                v.len()
            ))),
        }
    }

    pub fn linear(&self, name: &str, out_dim: usize, in_dim: usize) -> Result<Linear<B::Tensor>> {
        Ok(Linear {
            w: self.get(&format!("{name}.weight"), &[out_dim, in_dim])?,
            b: None,
        })
    }

    pub fn linear_bias(
        &self,
        name: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Linear<B::Tensor>> {
        Ok(Linear {
            w: self.get(&format!("{name}.weight"), &[out_dim, in_dim])?,
            b: Some(self.get(&format!("{name}.bias"), &[out_dim])?),
        })
    }

    /// `<name>.linear.weight` plus the optional `input_min/max`,
    /// `output_min/max` scalars of a `ClippableLinear`.
    pub fn clipped(
        &self,
        name: &str,
        out_dim: usize,
        in_dim: usize,
        clipped: bool,
    ) -> Result<ClippedLinear<B::Tensor>> {
        let linear = self.linear(&format!("{name}.linear"), out_dim, in_dim)?;
        let bound = |side: &str| -> Result<Option<(f64, f64)>> {
            let (lo, hi) = (format!("{name}.{side}_min"), format!("{name}.{side}_max"));
            if !clipped || !self.has(&lo) || !self.has(&hi) {
                return Ok(None);
            }
            let (lo, hi) = (self.scalar(&lo)?, self.scalar(&hi)?);
            Ok((lo.is_finite() || hi.is_finite()).then_some((lo, hi)))
        };
        Ok(ClippedLinear {
            input: bound("input")?,
            output: bound("output")?,
            linear,
        })
    }
}

/// Round an f32 to the backend precision and back — for scalars that
/// `transformers` keeps in the weight dtype (the √hidden embedding scale).
pub fn round_to<B: Backend>(be: &B, x: f32, precision: Precision) -> Result<f64> {
    let t = be.cast(&be.f32_tensor(&[x], &[1])?, precision)?;
    Ok(be.to_f32_vec(&t)?[0] as f64)
}

/// One preprocessed media item, on the host.
#[derive(Debug, Clone)]
pub enum MediaInput {
    Image(ImagePatches),
    Audio(AudioFeatures),
}

/// One input sequence: token ids (with placeholder runs for media soft
/// tokens) plus the media, in placeholder order.
#[derive(Debug, Clone, Default)]
pub struct PreparedRow {
    pub ids: Vec<u32>,
    pub media: Vec<MediaInput>,
}

/// The type-erased model the [`crate::Embedder`] drives.
pub trait Engine: Send + Sync {
    fn device_name(&self) -> String;
    fn precision(&self) -> Precision;
    fn modalities(&self) -> Modalities;
    fn embedding_dim(&self) -> usize;
    /// Mean-pooled (not normalized) vectors, one per row, running the rows
    /// in the given groups.
    fn embed_rows(&self, rows: &[PreparedRow], batches: &[Vec<usize>]) -> Result<Vec<Vec<f32>>>;
}

/// The composite model.
pub struct EmbeddingGemma2<B: Backend> {
    be: B,
    cfg: ModelConfig,
    text: text::TextModel<B>,
    vision: Option<vision::VisionTower<B>>,
    audio: Option<audio::AudioTower<B>>,
}

impl<B: Backend> EmbeddingGemma2<B> {
    /// Load the towers `modalities` asks for (the text model always).
    /// `max_positions` bounds the RoPE tables.
    pub fn load(
        be: B,
        cfg: ModelConfig,
        paths: &[PathBuf],
        modalities: Modalities,
        max_positions: usize,
    ) -> Result<Self> {
        let weights = be.open_weights(paths)?;
        let loader = Loader {
            be: &be,
            weights: &weights,
        };
        let text = text::TextModel::load(&loader, &cfg.text_config, max_positions)?;
        let vision = if modalities.image {
            let vc = cfg.vision_config.as_ref().ok_or_else(|| {
                EmbedError::Config(
                    "image input requested but config.json has no vision_config".into(),
                )
            })?;
            Some(vision::VisionTower::load(
                &loader,
                vc,
                cfg.text_config.hidden_size,
            )?)
        } else {
            None
        };
        let audio = if modalities.audio {
            let ac = cfg.audio_config.as_ref().ok_or_else(|| {
                EmbedError::Config(
                    "audio input requested but config.json has no audio_config".into(),
                )
            })?;
            Some(audio::AudioTower::load(
                &loader,
                ac,
                cfg.text_config.hidden_size,
            )?)
        } else {
            None
        };
        drop(weights);
        Ok(Self {
            be,
            cfg,
            text,
            vision,
            audio,
        })
    }

    fn is_placeholder(&self, id: u32) -> bool {
        id == self.cfg.image_token_id
            || id == self.cfg.video_token_id
            || id == self.cfg.audio_token_id
    }

    /// Soft tokens `[n, hidden]` for one media item.
    fn encode_media(&self, media: &MediaInput) -> Result<B::Tensor> {
        match media {
            MediaInput::Image(p) => self
                .vision
                .as_ref()
                .ok_or_else(|| {
                    EmbedError::InvalidRequest(
                        "this model was loaded without the vision tower (image / video input)"
                            .into(),
                    )
                })?
                .forward(&self.be, p),
            MediaInput::Audio(a) => self
                .audio
                .as_ref()
                .ok_or_else(|| {
                    EmbedError::InvalidRequest(
                        "this model was loaded without the audio tower (audio input)".into(),
                    )
                })?
                .forward(&self.be, a),
        }
    }

    /// Input embeddings `[1, seq_len, hidden]` of one row padded to
    /// `seq_len`: scaled token embeddings with the media soft tokens
    /// (unscaled) at the placeholder positions.
    fn row_embeddings(&self, row: &PreparedRow, seq_len: usize) -> Result<B::Tensor> {
        let be = &self.be;
        let pad = self.cfg.text_config.pad_id();
        let mut ids: Vec<u32> = row
            .ids
            .iter()
            .map(|&id| if self.is_placeholder(id) { pad } else { id })
            .collect();
        ids.resize(seq_len, pad);
        let tokens = self
            .text
            .embed_tokens(be, &be.u32_tensor(&ids, &[1, seq_len])?)?;
        if row.media.is_empty() {
            return Ok(tokens);
        }
        let hidden = self.cfg.text_config.hidden_size;
        let mut pieces = Vec::new();
        let mut media = row.media.iter();
        let mut i = 0;
        let n = row.ids.len();
        while i < n {
            let start = i;
            if self.is_placeholder(row.ids[i]) {
                while i < n && self.is_placeholder(row.ids[i]) {
                    i += 1;
                }
                let item = media.next().ok_or_else(|| {
                    EmbedError::InvalidRequest("more placeholder runs than media items".into())
                })?;
                let feats = self.encode_media(item)?;
                let rows = be.shape(&feats)[0];
                if rows != i - start {
                    return Err(EmbedError::InvalidRequest(format!(
                        "media produced {rows} soft tokens for {} placeholders",
                        i - start
                    )));
                }
                pieces.push(be.reshape(&be.cast(&feats, be.precision())?, &[1, rows, hidden])?);
            } else {
                while i < n && !self.is_placeholder(row.ids[i]) {
                    i += 1;
                }
                pieces.push(be.narrow(&tokens, 1, start, i - start)?);
            }
        }
        if media.next().is_some() {
            return Err(EmbedError::InvalidRequest(
                "more media items than placeholder runs".into(),
            ));
        }
        if seq_len > n {
            pieces.push(be.narrow(&tokens, 1, n, seq_len - n)?);
        }
        be.cat(&pieces, 1)
    }

    fn forward_batch(&self, rows: &[&PreparedRow]) -> Result<Vec<Vec<f32>>> {
        let be = &self.be;
        let lengths: Vec<usize> = rows.iter().map(|r| r.ids.len().max(1)).collect();
        let seq_len = *lengths.iter().max().unwrap_or(&1);
        let embeds = rows
            .iter()
            .map(|r| self.row_embeddings(r, seq_len))
            .collect::<Result<Vec<_>>>()?;
        let embeds = if embeds.len() == 1 {
            embeds.into_iter().next().expect("one row")
        } else {
            be.cat(&embeds, 0)?
        };
        let hidden = self.text.forward(be, &embeds, &lengths)?; // [b, s, 768]
        // sentence-transformers Pooling(mean): every real token, prompt and
        // special tokens included.
        let mut weights = Vec::with_capacity(rows.len() * seq_len);
        for &len in &lengths {
            weights.extend((0..seq_len).map(|i| if i < len { 1.0 / len as f32 } else { 0.0 }));
        }
        let weights = be.f32_tensor(&weights, &[rows.len(), seq_len, 1])?;
        let hidden = be.cast(&hidden, Precision::F32)?;
        let pooled = be.sum(&be.mul(&hidden, &weights)?, 1, false)?; // [b, 768]
        let flat = be.to_f32_vec(&pooled)?;
        let dim = self.text.embedding_dim();
        Ok(flat.chunks(dim).map(<[f32]>::to_vec).collect())
    }
}

impl<B: Backend> Engine for EmbeddingGemma2<B> {
    fn device_name(&self) -> String {
        self.be.device_name()
    }

    fn precision(&self) -> Precision {
        self.be.precision()
    }

    fn modalities(&self) -> Modalities {
        Modalities {
            image: self.vision.is_some(),
            audio: self.audio.is_some(),
        }
    }

    fn embedding_dim(&self) -> usize {
        self.text.embedding_dim()
    }

    fn embed_rows(&self, rows: &[PreparedRow], batches: &[Vec<usize>]) -> Result<Vec<Vec<f32>>> {
        let mut out = vec![Vec::new(); rows.len()];
        for batch in batches {
            let group: Vec<&PreparedRow> = batch.iter().map(|&i| &rows[i]).collect();
            for (&i, v) in batch.iter().zip(self.forward_batch(&group)?) {
                out[i] = v;
            }
        }
        Ok(out)
    }
}
