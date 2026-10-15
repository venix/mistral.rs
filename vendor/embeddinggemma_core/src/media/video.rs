//! Video and PDF inputs, both reduced to images for the vision tower.
//!
//! - **Video** (`EmbeddingGemma2VideoProcessor`): frames sampled at 1 fps,
//!   at most 32 (evenly spaced when there are more), 140 soft tokens per
//!   frame. Containers (MP4, MOV, WebM, …) are decoded by an external
//!   `ffmpeg`; clients without one can send the frames themselves.
//! - **PDF**: pages rendered by an external `pdftoppm` (poppler) and embedded
//!   as page images, the way the model card describes documents.
//!
//! Neither tool is linked in; when one is missing the request fails with a
//! message naming it.

use crate::error::{EmbedError, Result};
use image::RgbImage;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// External tools and sampling parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOptions {
    /// `ffmpeg` executable (name on `PATH` or a path).
    pub ffmpeg: String,
    /// `pdftoppm` executable (name on `PATH` or a path).
    pub pdftoppm: String,
    pub video_fps: f64,
    pub video_max_frames: usize,
    pub pdf_max_pages: usize,
    pub pdf_dpi: u32,
}

impl Default for ToolOptions {
    fn default() -> Self {
        Self {
            ffmpeg: "ffmpeg".into(),
            pdftoppm: "pdftoppm".into(),
            video_fps: 1.0,
            video_max_frames: 32,
            pdf_max_pages: 6,
            pdf_dpi: 144,
        }
    }
}

/// A scratch directory removed on drop.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let dir =
            std::env::temp_dir().join(format!("embeddinggemma-{}-{n}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(tool: &str, cmd: &mut Command, what: &str, env_var: &str) -> Result<()> {
    let output = cmd.output().map_err(|e| {
        EmbedError::Media(format!(
            "{what} input needs `{tool}` ({e}); install it or point {env_var} at it"
        ))
    })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(EmbedError::Media(format!(
            "`{tool}` failed on the {what}: {}",
            stderr.trim().lines().last().unwrap_or("unknown error")
        )));
    }
    Ok(())
}

fn read_images(dir: &Path, extension: &str) -> Result<Vec<RgbImage>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|x| x.eq_ignore_ascii_case(extension))
        })
        .collect();
    files.sort();
    files
        .iter()
        .map(|p| {
            image::open(p)
                .map(|i| i.to_rgb8())
                .map_err(|e| EmbedError::Media(format!("cannot read {}: {e}", p.display())))
        })
        .collect()
}

/// `overflow_strategy = "uniform"`: `np.linspace(0, n - 1, max, dtype=int)`.
pub fn uniform_indices(n: usize, max: usize) -> Vec<usize> {
    if n <= max || max == 0 {
        return (0..n).collect();
    }
    if max == 1 {
        return vec![0];
    }
    (0..max)
        .map(|i| ((i as f64) * ((n - 1) as f64) / ((max - 1) as f64)) as usize)
        .collect()
}

/// Keep at most `max` frames, evenly spaced.
pub fn cap_frames(frames: Vec<RgbImage>, max: usize) -> Vec<RgbImage> {
    let keep = uniform_indices(frames.len(), max);
    if keep.len() == frames.len() {
        return frames;
    }
    let mut frames: Vec<Option<RgbImage>> = frames.into_iter().map(Some).collect();
    keep.into_iter().filter_map(|i| frames[i].take()).collect()
}

/// Decode a video container into frames sampled at `video_fps`, capped at
/// `video_max_frames`.
pub fn decode_video(
    bytes: &[u8],
    extension: Option<&str>,
    opts: &ToolOptions,
) -> Result<Vec<RgbImage>> {
    let dir = ScratchDir::new()?;
    let input = dir.0.join(format!("input.{}", extension.unwrap_or("mp4")));
    std::fs::write(&input, bytes)?;
    let pattern = dir.0.join("frame_%06d.png");
    run(
        &opts.ffmpeg,
        Command::new(&opts.ffmpeg)
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(&input)
            .args(["-an", "-vf", &format!("fps={}", opts.video_fps)])
            // Bound the work for very long inputs; the cap below thins further.
            .args(["-frames:v", "3600"])
            .arg(&pattern),
        "video",
        "EMBEDDINGGEMMA_FFMPEG",
    )?;
    let frames = read_images(&dir.0, "png")?;
    if frames.is_empty() {
        return Err(EmbedError::Media("video has no decodable frames".into()));
    }
    Ok(cap_frames(frames, opts.video_max_frames))
}

/// Render the first `pdf_max_pages` pages of a PDF at `pdf_dpi`.
pub fn render_pdf(bytes: &[u8], opts: &ToolOptions) -> Result<Vec<RgbImage>> {
    let dir = ScratchDir::new()?;
    let input = dir.0.join("input.pdf");
    std::fs::write(&input, bytes)?;
    run(
        &opts.pdftoppm,
        Command::new(&opts.pdftoppm)
            .args(["-png", "-r", &opts.pdf_dpi.to_string()])
            .args(["-f", "1", "-l", &opts.pdf_max_pages.max(1).to_string()])
            .arg(&input)
            .arg(dir.0.join("page")),
        "PDF",
        "EMBEDDINGGEMMA_PDFTOPPM",
    )?;
    let pages = read_images(&dir.0, "png")?;
    if pages.is_empty() {
        return Err(EmbedError::Media("PDF has no renderable pages".into()));
    }
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_sampling_matches_numpy_linspace() {
        assert_eq!(uniform_indices(5, 32), vec![0, 1, 2, 3, 4]);
        // np.linspace(0, 99, 4, dtype=int) = [0, 33, 66, 99]
        assert_eq!(uniform_indices(100, 4), vec![0, 33, 66, 99]);
        // np.linspace(0, 9, 3, dtype=int) = [0, 4, 9]
        assert_eq!(uniform_indices(10, 3), vec![0, 4, 9]);
        let frames: Vec<RgbImage> = (0..10).map(|i| RgbImage::new(i + 1, 1)).collect();
        let kept = cap_frames(frames, 3);
        assert_eq!(
            kept.iter().map(|f| f.width()).collect::<Vec<_>>(),
            vec![1, 5, 10]
        );
    }

    #[test]
    fn missing_tools_are_reported_by_name() {
        let opts = ToolOptions {
            ffmpeg: "definitely-not-ffmpeg-xyz".into(),
            ..Default::default()
        };
        let e = decode_video(b"x", Some("mp4"), &opts)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("definitely-not-ffmpeg-xyz") && e.contains("EMBEDDINGGEMMA_FFMPEG"),
            "{e}"
        );
    }
}
