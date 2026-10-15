//! Multimodal content items — wire-compatible with GAISe's `GaiseContent`
//! (`{"type": "text" | "image" | "audio" | "file" | "parts", …}`), plus a
//! `video` item.
//!
//! Binary `data` is accepted as GAISe serialises it (a JSON array of bytes),
//! as a base64 string, or as a `data:` URL; it is serialised back as an
//! array of bytes so GAISe can read it.

use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD, URL_SAFE};
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::{Deserialize, Serialize, Serializer};
use std::fmt;

/// Raw bytes with lenient JSON decoding.
#[derive(Clone, PartialEq, Eq, Default)]
pub struct Bytes {
    pub data: Vec<u8>,
    /// The MIME type of a `data:` URL, when one was given.
    pub mime: Option<String>,
}

impl Bytes {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data, mime: None }
    }
}

impl fmt::Debug for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bytes({} bytes", self.data.len())?;
        if let Some(m) = &self.mime {
            write!(f, ", {m}")?;
        }
        write!(f, ")")
    }
}

impl From<Vec<u8>> for Bytes {
    fn from(data: Vec<u8>) -> Self {
        Self::new(data)
    }
}

impl Serialize for Bytes {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(&self.data)
    }
}

/// Decode base64 (standard or URL-safe, padding optional) or a
/// `data:<mime>;base64,<payload>` URL.
pub fn decode_base64(value: &str) -> Result<Bytes, String> {
    let value = value.trim();
    let (mime, payload) = match value.strip_prefix("data:") {
        Some(rest) => {
            let (meta, payload) = rest
                .split_once(',')
                .ok_or_else(|| "malformed data URL (no comma)".to_string())?;
            if !meta.ends_with(";base64") {
                return Err("only base64 data URLs are supported".into());
            }
            let mime = meta.trim_end_matches(";base64");
            (Some(mime.to_string()).filter(|m| !m.is_empty()), payload)
        }
        None => (None, value),
    };
    let cleaned: String = payload.chars().filter(|c| !c.is_whitespace()).collect();
    let padded = match cleaned.len() % 4 {
        2 => format!("{cleaned}=="),
        3 => format!("{cleaned}="),
        _ => cleaned,
    };
    let data = STANDARD
        .decode(&padded)
        .or_else(|_| URL_SAFE.decode(&padded))
        .map_err(|e| format!("invalid base64: {e}"))?;
    Ok(Bytes { data, mime })
}

impl<'de> Deserialize<'de> for Bytes {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Bytes;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an array of bytes, a base64 string, or a data: URL")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Bytes, E> {
                decode_base64(v).map_err(E::custom)
            }

            fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Bytes, E> {
                Ok(Bytes::new(v.to_vec()))
            }

            fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Bytes, E> {
                Ok(Bytes::new(v))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Bytes, A::Error> {
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(b) = seq.next_element::<u8>()? {
                    out.push(b);
                }
                Ok(Bytes::new(out))
            }
        }
        d.deserialize_any(V)
    }
}

/// One content item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
    },
    /// PNG, JPEG, WebP, GIF or BMP.
    Image {
        data: Bytes,
        /// `png`, `image/png`, … (optional; detected from the bytes).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<String>,
    },
    /// WAV, MP3, FLAC, Ogg/Vorbis or M4A/AAC; resampled to 16 kHz mono.
    Audio {
        data: Bytes,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<String>,
    },
    /// Extension (not in GAISe): a video container in `data` (decoded with
    /// `ffmpeg`), or already-extracted `frames` (images, in order).
    Video {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<Bytes>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        frames: Vec<Bytes>,
    },
    /// Any file; routed by `name` / MIME / magic bytes to image, audio,
    /// video, PDF (pages as images) or UTF-8 text.
    File {
        data: Bytes,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// Several parts embedded together into one vector.
    Parts {
        parts: Vec<Content>,
    },
}

impl Default for Content {
    fn default() -> Self {
        Self::Text {
            text: String::new(),
        }
    }
}

impl Content {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Whether this item (recursively) holds only text.
    pub fn is_text_only(&self) -> bool {
        match self {
            Self::Text { .. } => true,
            Self::Parts { parts } => parts.iter().all(Content::is_text_only),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_gaise_byte_arrays_base64_and_data_urls() {
        let a: Content =
            serde_json::from_str(r#"{"type":"image","data":[1,2,3],"format":"png"}"#).unwrap();
        let b: Content =
            serde_json::from_str(r#"{"type":"image","data":"AQID","format":"png"}"#).unwrap();
        assert_eq!(a, b);
        let c: Content =
            serde_json::from_str(r#"{"type":"image","data":"data:image/png;base64,AQID"}"#)
                .unwrap();
        match c {
            Content::Image { data, .. } => {
                assert_eq!(data.data, vec![1, 2, 3]);
                assert_eq!(data.mime.as_deref(), Some("image/png"));
            }
            other => panic!("{other:?}"),
        }
        // GAISe round trip: bytes serialise as an array.
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            r#"{"type":"image","data":[1,2,3],"format":"png"}"#
        );
        assert_eq!(decode_base64("AQI").unwrap().data, vec![1, 2]);
        assert!(decode_base64("data:text/plain,hi").is_err());
    }

    #[test]
    fn parses_parts_files_and_video() {
        let json = r#"{"type":"parts","parts":[{"type":"text","text":"a cat"},{"type":"file","data":"JVBERg==","name":"x.pdf"}]}"#;
        let c: Content = serde_json::from_str(json).unwrap();
        assert!(!c.is_text_only());
        assert!(
            Content::Parts {
                parts: vec![Content::text("a"), Content::text("b")]
            }
            .is_text_only()
        );
        let v: Content =
            serde_json::from_str(r#"{"type":"video","frames":["AQID","BAUG"]}"#).unwrap();
        assert!(matches!(v, Content::Video { data: None, ref frames, .. } if frames.len() == 2));
        assert!(serde_json::from_str::<Content>(r#"{"type":"reasoning","text":"x"}"#).is_err());
    }
}
