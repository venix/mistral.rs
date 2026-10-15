//! Known checkpoints, their names, and which towers to load.
//!
//! EmbeddingGemma 2 is one 740M-parameter checkpoint: a 270M text encoder
//! plus a 167M vision tower and a 305M audio tower. The published sizes are
//! subsets of it — load only the towers a deployment needs:
//!
//! | Name | Towers | Inputs |
//! |---|---|---|
//! | `embeddinggemma-2` / `740m` | text + vision + audio | text, image, video, PDF, audio |
//! | `440m` | text + vision | text, image, video, PDF |
//! | `570m` | text + audio | text, audio |
//! | `270m` | text | text |

use serde::Serialize;
use std::fmt;

/// The provider key used in `provider::model` ids.
pub const PROVIDER: &str = "embeddinggemma";

/// Which optional towers are loaded (the text encoder always is). Video and
/// PDF go through the vision tower.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Modalities {
    pub image: bool,
    pub audio: bool,
}

impl Modalities {
    pub const ALL: Self = Self {
        image: true,
        audio: true,
    };
    pub const TEXT: Self = Self {
        image: false,
        audio: false,
    };

    /// `text,image,audio`, `all`, `text` (`video` / `pdf` imply `image`).
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut m = Self::TEXT;
        for part in s
            .split([',', '+', ' '])
            .map(|p| p.trim().to_ascii_lowercase())
        {
            match part.as_str() {
                "" | "text" => {}
                "all" => m = Self::ALL,
                "image" | "images" | "vision" | "video" | "pdf" => m.image = true,
                "audio" => m.audio = true,
                other => {
                    return Err(format!(
                        "unknown modality `{other}` (expected text, image, video, pdf, audio or all)"
                    ));
                }
            }
        }
        Ok(m)
    }

    /// Input kinds accepted, for logs and `/v1/models` (`capabilities.input`).
    pub fn inputs(&self) -> Vec<&'static str> {
        let mut out = vec!["text"];
        if self.image {
            out.extend(["image", "video", "pdf"]);
        }
        if self.audio {
            out.push("audio");
        }
        out
    }

    /// Parameter count of this subset of EmbeddingGemma 2.
    pub fn parameters(&self) -> &'static str {
        match (self.image, self.audio) {
            (true, true) => "740M",
            (true, false) => "440M",
            (false, true) => "570M",
            (false, false) => "270M",
        }
    }
}

/// Serialised as the input kinds it accepts (`["text", "image", …]`).
impl Serialize for Modalities {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.inputs().serialize(s)
    }
}

impl fmt::Display for Modalities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = vec!["text"];
        if self.image {
            parts.push("image");
        }
        if self.audio {
            parts.push("audio");
        }
        f.write_str(&parts.join(","))
    }
}

/// One known checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelSpec {
    /// Bare id, e.g. `embeddinggemma-2`.
    pub id: &'static str,
    /// Hugging Face repo, e.g. `google/embeddinggemma-2`.
    pub repo: &'static str,
    pub display_name: &'static str,
    pub embedding_dimensions: u32,
    /// Matryoshka sizes the model was trained for (truncate + re-normalize).
    pub matryoshka_dimensions: &'static [u32],
    /// Documented context, shared by every modality.
    pub max_input_tokens: u32,
    /// Approximate weight size in bf16, all towers.
    pub weights_gb: f32,
    pub aliases: &'static [&'static str],
}

impl ModelSpec {
    pub fn qualified_id(&self) -> String {
        format!("{PROVIDER}::{}", self.id)
    }
}

pub const MODELS: &[ModelSpec] = &[ModelSpec {
    id: "embeddinggemma-2",
    repo: "google/embeddinggemma-2",
    display_name: "EmbeddingGemma 2",
    embedding_dimensions: 768,
    matryoshka_dimensions: &[768, 512, 256, 128],
    max_input_tokens: 8192,
    weights_gb: 1.49,
    aliases: &[
        "embeddinggemma2",
        "embedding-gemma-2",
        "eg2",
        "gemma-embedding-2",
    ],
}];

/// The default model when the operator configures none.
pub const DEFAULT_MODEL_ID: &str = "embeddinggemma-2";

/// Size suffixes that select a subset of towers.
const VARIANTS: &[(&str, Modalities)] = &[
    ("740m", Modalities::ALL),
    (
        "570m",
        Modalities {
            image: false,
            audio: true,
        },
    ),
    (
        "440m",
        Modalities {
            image: true,
            audio: false,
        },
    ),
    ("270m", Modalities::TEXT),
];

/// Strip a `provider::` prefix (any provider — a GAISe router may forward
/// `embeddinggemma::x`, a user may type `x`).
pub fn bare_id(name: &str) -> &str {
    let name = name.trim();
    match name.split_once("::") {
        Some((_, rest)) => rest.trim(),
        None => name,
    }
}

/// Find a known checkpoint by qualified id, bare id, Hub repo, alias, or
/// size variant (`270m`, `embeddinggemma-2-440m`), case-insensitively. The
/// second value is the tower subset a size variant implies.
pub fn resolve(name: &str) -> Option<(&'static ModelSpec, Option<Modalities>)> {
    let bare = bare_id(name).to_ascii_lowercase();
    if bare.is_empty() {
        return None;
    }
    let matches = |m: &ModelSpec, s: &str| {
        m.id == s || m.repo.to_ascii_lowercase() == s || m.aliases.contains(&s)
    };
    for m in MODELS {
        if matches(m, &bare) {
            return Some((m, None));
        }
        for (size, modalities) in VARIANTS {
            if bare == *size {
                return Some((m, Some(*modalities)));
            }
            if let Some(base) = bare
                .strip_suffix(size)
                .map(|b| b.trim_end_matches(['-', '_', ':']))
                && matches(m, base)
            {
                return Some((m, Some(*modalities)));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_every_spelling() {
        for name in [
            "embeddinggemma-2",
            "embeddinggemma::embeddinggemma-2",
            "google/embeddinggemma-2",
            "Google/EmbeddingGemma-2",
            "eg2",
            "ollama::embeddinggemma-2",
        ] {
            assert_eq!(
                resolve(name).map(|(m, v)| (m.id, v)),
                Some(("embeddinggemma-2", None)),
                "{name}"
            );
        }
        assert_eq!(resolve("270m").unwrap().1, Some(Modalities::TEXT));
        assert_eq!(
            resolve("embeddinggemma-2-440m").unwrap().1.unwrap(),
            Modalities {
                image: true,
                audio: false
            }
        );
        assert_eq!(
            resolve("embeddinggemma-2:570m").unwrap().1.unwrap(),
            Modalities {
                image: false,
                audio: true
            }
        );
        assert_eq!(resolve("740m").unwrap().1, Some(Modalities::ALL));
        assert_eq!(resolve("nope"), None);
        assert_eq!(resolve(""), None);
    }

    #[test]
    fn parses_modalities() {
        assert_eq!(Modalities::parse("all").unwrap(), Modalities::ALL);
        assert_eq!(Modalities::parse("text").unwrap(), Modalities::TEXT);
        assert_eq!(
            Modalities::parse("text,video").unwrap(),
            Modalities {
                image: true,
                audio: false
            }
        );
        assert_eq!(Modalities::parse("audio").unwrap().parameters(), "570M");
        assert!(Modalities::parse("smell").is_err());
        assert_eq!(Modalities::ALL.to_string(), "text,image,audio");
        assert_eq!(
            Modalities::ALL.inputs(),
            vec!["text", "image", "video", "pdf", "audio"]
        );
    }
}
