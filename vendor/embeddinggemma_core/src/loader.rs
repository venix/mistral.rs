//! Locating model files: a local directory or a Hugging Face Hub repo
//! (downloaded once into the Hub cache, `HF_HOME` / `HF_TOKEN` honoured).
//!
//! The recommended layout for deployment is a flat **model directory** next
//! to the project (`./models/embeddinggemma-2/config.json`, `tokenizer.json`,
//! `model.safetensors`, `processor_config.json`, …), created with
//! [`export_to_dir`] / `embeddinggemma_api download`. [`resolve_source`] checks
//! it before going to the Hub, so a populated directory means no network at
//! startup — which is what a Docker image or an air-gapped box wants.

use crate::error::{EmbedError, Result};
use crate::registry;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Environment variable naming the model directory; default `./models`.
pub const MODEL_DIR_ENV: &str = "EMBEDDINGGEMMA_MODEL_DIR";

/// The model directory: `$EMBEDDINGGEMMA_MODEL_DIR`, else `./models`.
pub fn model_dir() -> PathBuf {
    std::env::var_os(MODEL_DIR_ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"))
}

/// Where a checkpoint comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// A Hugging Face repo such as `google/embeddinggemma-2`.
    Hub {
        repo: String,
        revision: Option<String>,
    },
    /// A directory holding `config.json`, `tokenizer.json` and safetensors.
    Local(PathBuf),
}

impl ModelSource {
    /// A directory that exists is local; anything else is a Hub repo id
    /// (`owner/name` or `owner/name@revision`).
    pub fn parse(value: &str) -> Self {
        let value = value.trim();
        if Path::new(value).is_dir() {
            return Self::Local(PathBuf::from(value));
        }
        match value.rsplit_once('@') {
            Some((repo, rev)) if !repo.contains('@') && !rev.is_empty() => Self::Hub {
                repo: repo.to_string(),
                revision: Some(rev.to_string()),
            },
            _ => Self::Hub {
                repo: value.to_string(),
                revision: None,
            },
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Hub {
                repo,
                revision: None,
            } => repo.clone(),
            Self::Hub {
                repo,
                revision: Some(r),
            } => format!("{repo}@{r}"),
            Self::Local(p) => p.display().to_string(),
        }
    }
}

/// Turn a model name into a source, preferring local copies:
///
/// 1. a directory path as given;
/// 2. `<model_dir>/<id>` (registry ids, aliases and size variants map to the
///    registry id) when it holds a `config.json`;
/// 3. `<model_dir>/<last repo segment>` for an unregistered `owner/name`;
/// 4. the registry's Hub repo, else the name as a Hub repo id.
pub fn resolve_source(name: &str, model_dir: &Path) -> ModelSource {
    let bare = registry::bare_id(name);
    if Path::new(bare).is_dir() {
        return ModelSource::Local(PathBuf::from(bare));
    }
    let spec = registry::resolve(bare).map(|(s, _)| s);
    let local_id = spec.map(|s| s.id).unwrap_or(bare);
    let candidate = model_dir.join(local_id);
    if candidate.join("config.json").is_file() {
        return ModelSource::Local(candidate);
    }
    if bare.contains('/') {
        let last = bare.rsplit('/').next().unwrap_or(bare);
        let candidate = model_dir.join(last.split('@').next().unwrap_or(last));
        if candidate.join("config.json").is_file() {
            return ModelSource::Local(candidate);
        }
    }
    match spec {
        Some(spec) => ModelSource::Hub {
            repo: spec.repo.to_string(),
            revision: None,
        },
        None => ModelSource::parse(bare),
    }
}

/// Bare id for an unregistered source: last path segment of the repo or the
/// directory name.
pub fn source_id(source: &ModelSource) -> String {
    match source {
        ModelSource::Hub { repo, .. } => repo.rsplit('/').next().unwrap_or(repo).to_string(),
        ModelSource::Local(path) => path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string()),
    }
}

/// The files needed to load one model.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    /// Safetensors shards in index order.
    pub weights: Vec<PathBuf>,
    /// `processor_config.json` (image / video / audio preprocessing).
    pub processor_config: Option<PathBuf>,
    /// `config_sentence_transformers.json` (task prompts).
    pub sentence_transformers_config: Option<PathBuf>,
    /// Other files worth carrying along when exporting, as
    /// `(relative name, path)`.
    pub extras: Vec<(String, PathBuf)>,
}

/// Optional companions copied by [`export_to_dir`] (relative paths).
const EXTRA_FILES: [&str; 9] = [
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "2_Normalize/config.json",
    "model.safetensors.index.json",
    "README.md",
];

impl ModelFiles {
    /// Locate (and for Hub sources, download) the files.
    pub fn locate(source: &ModelSource) -> Result<Self> {
        match source {
            ModelSource::Local(dir) => Self::from_dir(dir),
            ModelSource::Hub { repo, revision } => Self::from_hub(repo, revision.as_deref()),
        }
    }

    fn from_dir(dir: &Path) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        let weights = if index.is_file() {
            shard_names(&std::fs::read_to_string(&index)?)?
                .into_iter()
                .map(|n| existing(dir.join(n)))
                .collect::<Result<Vec<_>>>()?
        } else if dir.join("model.safetensors").is_file() {
            vec![dir.join("model.safetensors")]
        } else {
            let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                .collect();
            shards.sort();
            if shards.is_empty() {
                return Err(EmbedError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("no .safetensors files in {}", dir.display()),
                )));
            }
            shards
        };
        let optional = |name: &str| Some(dir.join(name)).filter(|p| p.is_file());
        Ok(Self {
            config: existing(dir.join("config.json"))?,
            tokenizer: existing(dir.join("tokenizer.json"))?,
            weights,
            processor_config: optional("processor_config.json"),
            sentence_transformers_config: optional("config_sentence_transformers.json"),
            extras: EXTRA_FILES
                .iter()
                .filter_map(|f| optional(f).map(|p| (f.to_string(), p)))
                .collect(),
        })
    }

    fn from_hub(repo: &str, revision: Option<&str>) -> Result<Self> {
        let weights = match hub_file(repo, revision, "model.safetensors.index.json") {
            Ok(index) => shard_names(&std::fs::read_to_string(&index)?)?
                .into_iter()
                .map(|n| hub_file(repo, revision, &n))
                .collect::<Result<Vec<_>>>()?,
            Err(_) => vec![hub_file(repo, revision, "model.safetensors")?],
        };
        Ok(Self {
            config: hub_file(repo, revision, "config.json")?,
            tokenizer: hub_file(repo, revision, "tokenizer.json")?,
            weights,
            processor_config: hub_file(repo, revision, "processor_config.json").ok(),
            sentence_transformers_config: hub_file(
                repo,
                revision,
                "config_sentence_transformers.json",
            )
            .ok(),
            extras: EXTRA_FILES
                .iter()
                .filter_map(|f| hub_file(repo, revision, f).ok().map(|p| (f.to_string(), p)))
                .collect(),
        })
    }
}

/// Fetch a model (from the Hub or another directory) and copy its files into
/// `<model_dir>/<id>/` as plain files — no symlinks, no cache layout — so the
/// directory can be mounted into a container or baked into an image. Returns
/// the directory. Existing files of the same size are kept.
pub fn export_to_dir(name: &str, model_dir: &Path) -> Result<PathBuf> {
    let bare = registry::bare_id(name);
    let spec = registry::resolve(bare).map(|(s, _)| s);
    let source = match spec {
        Some(spec) => ModelSource::Hub {
            repo: spec.repo.to_string(),
            revision: None,
        },
        None => ModelSource::parse(bare),
    };
    let id = spec
        .map(|s| s.id.to_string())
        .unwrap_or_else(|| source_id(&source));
    let files = ModelFiles::locate(&source)?;
    let target = model_dir.join(&id);
    std::fs::create_dir_all(&target)?;

    let mut copies: Vec<(String, PathBuf)> = vec![
        ("config.json".into(), files.config.clone()),
        ("tokenizer.json".into(), files.tokenizer.clone()),
    ];
    for p in &files.weights {
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .ok_or_else(|| EmbedError::Io(std::io::Error::other("weight path has no file name")))?;
        copies.push((name, p.clone()));
    }
    if let Some(p) = &files.processor_config {
        copies.push(("processor_config.json".into(), p.clone()));
    }
    if let Some(p) = &files.sentence_transformers_config {
        copies.push(("config_sentence_transformers.json".into(), p.clone()));
    }
    copies.extend(files.extras.iter().cloned());
    for (rel, src) in copies {
        let dst = target.join(&rel);
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let same = dst
            .metadata()
            .ok()
            .zip(src.metadata().ok())
            .is_some_and(|(d, s)| d.len() == s.len());
        if !same && dst != src {
            std::fs::copy(&src, &dst)?;
        }
    }
    Ok(target)
}

/// Download (or find in the cache) one file of a Hub repo.
#[cfg(feature = "hf-hub")]
fn hub_file(repo: &str, revision: Option<&str>, filename: &str) -> Result<PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Repo, RepoType};

    let hub = |e: hf_hub::api::sync::ApiError| EmbedError::Hub(format!("{repo}/{filename}: {e}"));
    let api = ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .map_err(hub)?;
    let handle = api.repo(match revision {
        Some(rev) => Repo::with_revision(repo.to_string(), RepoType::Model, rev.to_string()),
        None => Repo::new(repo.to_string(), RepoType::Model),
    });
    handle.get(filename).map_err(hub)
}

#[cfg(not(feature = "hf-hub"))]
fn hub_file(repo: &str, _revision: Option<&str>, filename: &str) -> Result<PathBuf> {
    Err(EmbedError::Hub(format!(
        "`{repo}/{filename}` is not available locally and this build has no `hf-hub` feature; \
         download the files and pass their directory"
    )))
}

fn existing(path: PathBuf) -> Result<PathBuf> {
    if path.is_file() {
        Ok(path)
    } else {
        Err(EmbedError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("missing model file {}", path.display()),
        )))
    }
}

/// Unique shard file names from a `model.safetensors.index.json`, in order.
fn shard_names(index_json: &str) -> Result<Vec<String>> {
    #[derive(serde::Deserialize)]
    struct Index {
        weight_map: std::collections::BTreeMap<String, String>,
    }
    let index: Index = serde_json::from_str(index_json)?;
    let names: BTreeSet<String> = index.weight_map.into_values().collect();
    if names.is_empty() {
        return Err(EmbedError::Config(
            "safetensors index has no weights".into(),
        ));
    }
    Ok(names.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sources() {
        assert_eq!(
            ModelSource::parse("google/embeddinggemma-2@abc"),
            ModelSource::Hub {
                repo: "google/embeddinggemma-2".into(),
                revision: Some("abc".into())
            }
        );
        let here = std::env::current_dir().unwrap();
        assert_eq!(
            ModelSource::parse(here.to_str().unwrap()),
            ModelSource::Local(here)
        );
    }

    #[test]
    fn prefers_the_model_dir() {
        let tmp =
            std::env::temp_dir().join(format!("embeddinggemma-modeldir-{}", std::process::id()));
        let local = tmp.join("embeddinggemma-2");
        std::fs::create_dir_all(&local).unwrap();
        assert!(matches!(
            resolve_source("eg2", &tmp),
            ModelSource::Hub { .. }
        ));
        std::fs::write(local.join("config.json"), "{}").unwrap();
        for name in [
            "embeddinggemma-2",
            "270m",
            "embeddinggemma::eg2",
            "google/embeddinggemma-2",
        ] {
            assert_eq!(
                resolve_source(name, &tmp),
                ModelSource::Local(local.clone()),
                "{name}"
            );
        }
        assert!(matches!(
            resolve_source("someone/other-model", &tmp),
            ModelSource::Hub { .. }
        ));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn reads_shard_index() {
        let json =
            r#"{"weight_map":{"a":"m-2.safetensors","b":"m-1.safetensors","c":"m-1.safetensors"}}"#;
        assert_eq!(
            shard_names(json).unwrap(),
            vec!["m-1.safetensors", "m-2.safetensors"]
        );
    }
}
