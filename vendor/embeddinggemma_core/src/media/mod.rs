//! Turning bytes into model inputs: image / audio / video / PDF decoding and
//! the EmbeddingGemma 2 processors, all on the host.

pub mod audio;
pub mod image;
pub mod video;

pub use video::ToolOptions;

/// One image (or video frame) cut into patches.
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePatches {
    /// Row-major patches (`index = y · patches_w + x`), each `p·p·3` values
    /// in `[0, 1]` with channels last.
    pub pixels: Vec<f32>,
    pub patches_h: usize,
    pub patches_w: usize,
    pub pooling_kernel_size: usize,
}

impl ImagePatches {
    /// Soft tokens after `k×k` pooling.
    pub fn soft_tokens(&self) -> usize {
        let k = self.pooling_kernel_size.max(1);
        (self.patches_h / k) * (self.patches_w / k)
    }
}

/// Log-mel features of one clip.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFeatures {
    /// `[frames, bins]`, row-major; invalid (padding) frames are zero.
    pub mel: Vec<f32>,
    pub frames: usize,
    pub bins: usize,
    /// Leading frames that are real audio.
    pub valid_frames: usize,
}

impl AudioFeatures {
    /// Soft tokens after the two stride-2 convolutions (40 ms each).
    pub fn soft_tokens(&self) -> usize {
        self.valid_frames.div_ceil(2).div_ceil(2)
    }
}

/// What a blob of bytes is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Audio,
    Video,
    Pdf,
    Text,
}

/// Classify by MIME type or file name first, then by magic bytes.
pub fn detect(bytes: &[u8], hint: Option<&str>) -> Option<MediaKind> {
    if let Some(kind) = hint.and_then(kind_from_hint) {
        return Some(kind);
    }
    sniff(bytes)
}

/// `image/png`, `png`, `report.pdf`, `clip.MP4`, … → kind.
pub fn kind_from_hint(hint: &str) -> Option<MediaKind> {
    let h = hint.trim().to_ascii_lowercase();
    if let Some((major, _)) = h.split_once('/')
        && !h.contains('.')
    {
        return match major {
            "image" => Some(MediaKind::Image),
            "audio" => Some(MediaKind::Audio),
            "video" => Some(MediaKind::Video),
            "text" => Some(MediaKind::Text),
            _ if h == "application/pdf" => Some(MediaKind::Pdf),
            _ if h == "application/json" => Some(MediaKind::Text),
            _ => None,
        };
    }
    let ext = h.rsplit('.').next().unwrap_or(&h);
    Some(match ext {
        "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" => MediaKind::Image,
        "wav" | "wave" | "mp3" | "mpeg" | "flac" | "ogg" | "oga" | "m4a" | "aac" | "opus" => {
            MediaKind::Audio
        }
        "mp4" | "mov" | "webm" | "mkv" | "avi" | "m4v" | "mpg" | "3gp" => MediaKind::Video,
        "pdf" => MediaKind::Pdf,
        "txt" | "md" | "markdown" | "csv" | "tsv" | "json" | "jsonl" | "html" | "htm" | "xml"
        | "yaml" | "yml" | "toml" | "log" | "rs" | "py" => MediaKind::Text,
        _ => return None,
    })
}

/// Magic-byte detection.
pub fn sniff(b: &[u8]) -> Option<MediaKind> {
    let at = |off: usize, sig: &[u8]| b.len() >= off + sig.len() && &b[off..off + sig.len()] == sig;
    if at(0, b"\x89PNG")
        || at(0, b"\xFF\xD8\xFF")
        || at(0, b"GIF8")
        || at(0, b"BM")
        || (at(0, b"RIFF") && at(8, b"WEBP"))
    {
        return Some(MediaKind::Image);
    }
    if at(0, b"%PDF") {
        return Some(MediaKind::Pdf);
    }
    if (at(0, b"RIFF") && at(8, b"WAVE"))
        || at(0, b"ID3")
        || at(0, b"fLaC")
        || at(0, b"OggS")
        || (b.len() > 1 && b[0] == 0xFF && (b[1] & 0xE0) == 0xE0)
    {
        return Some(MediaKind::Audio);
    }
    if at(4, b"ftyp") {
        return Some(if at(8, b"M4A ") || at(8, b"M4B ") {
            MediaKind::Audio
        } else {
            MediaKind::Video
        });
    }
    if at(0, b"\x1A\x45\xDF\xA3") || (at(0, b"RIFF") && at(8, b"AVI ")) {
        return Some(MediaKind::Video);
    }
    None
}

/// File extension for a MIME type or file name (helps the audio probe and
/// names temp files for `ffmpeg`).
pub fn extension_hint(hint: Option<&str>) -> Option<String> {
    let h = hint?.trim().to_ascii_lowercase();
    if h.contains('/') && !h.contains('.') {
        let sub = h.split('/').nth(1)?;
        return Some(
            match sub {
                "mpeg" | "mp3" => "mp3",
                "wave" | "x-wav" | "wav" => "wav",
                "quicktime" => "mov",
                "x-matroska" => "mkv",
                "mp4" | "m4a" => sub,
                "x-flac" => "flac",
                other => other,
            }
            .to_string(),
        );
    }
    h.rsplit_once('.').map(|(_, e)| e.to_string()).or(Some(h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_by_hint_and_magic() {
        assert_eq!(detect(b"", Some("image/png")), Some(MediaKind::Image));
        assert_eq!(detect(b"", Some("report.PDF")), Some(MediaKind::Pdf));
        assert_eq!(detect(b"", Some("clip.mov")), Some(MediaKind::Video));
        assert_eq!(detect(b"", Some("mp3")), Some(MediaKind::Audio));
        assert_eq!(detect(b"\x89PNG\r\n", None), Some(MediaKind::Image));
        assert_eq!(
            detect(b"RIFF\0\0\0\0WAVEfmt ", None),
            Some(MediaKind::Audio)
        );
        assert_eq!(detect(b"\0\0\0\x20ftypisom", None), Some(MediaKind::Video));
        assert_eq!(detect(b"\0\0\0\x20ftypM4A ", None), Some(MediaKind::Audio));
        assert_eq!(detect(b"%PDF-1.7", None), Some(MediaKind::Pdf));
        assert_eq!(detect(b"hello", None), None);
        assert_eq!(extension_hint(Some("audio/mpeg")).as_deref(), Some("mp3"));
        assert_eq!(extension_hint(Some("x.wav")).as_deref(), Some("wav"));
    }
}
