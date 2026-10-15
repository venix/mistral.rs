//! The embedding contract: the GAISe task vocabulary, EmbeddingGemma 2's
//! prompts, and the request / response wire types.
//!
//! The vocabulary, aliases and wire shapes mirror
//! `gaise_core::contracts::{GaiseEmbeddingTask, GaiseEmbeddingsRequest,
//! GaiseContentEmbeddingsRequest, GaiseEmbeddingsResponse}` so a GAISe client
//! (or router) can talk to this service unchanged. EmbeddingGemma 2 takes the
//! task as a text prefix (`task: search result | query: …`, `title: none |
//! text: …`); the prefixes come from the checkpoint's
//! `config_sentence_transformers.json`.

use crate::content::Content;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One value or an array (`"input": "x"` or `"input": ["x", "y"]`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> OneOrMany<T> {
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::One(x) => vec![x],
            Self::Many(v) => v,
        }
    }

    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::One(x) => std::slice::from_ref(x),
            Self::Many(v) => v,
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Default> Default for OneOrMany<T> {
    fn default() -> Self {
        Self::One(T::default())
    }
}

impl<T> From<T> for OneOrMany<T> {
    fn from(value: T) -> Self {
        Self::One(value)
    }
}

impl<T> From<Vec<T>> for OneOrMany<T> {
    fn from(value: Vec<T>) -> Self {
        Self::Many(value)
    }
}

/// What the vectors are for — the GAISe canonical task vocabulary.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum EmbeddingTask {
    /// Text being indexed for later retrieval (the default):
    /// `title: none | text: `.
    #[default]
    Document,
    /// A search query: `task: search result | query: `.
    Query,
    Classification,
    Clustering,
    /// Symmetric similarity: `task: sentence similarity | query: `.
    Similarity,
    /// A natural-language query for code retrieval.
    CodeQuery,
    FactVerification,
    QuestionAnswering,
    /// Anything else: a prompt name from the checkpoint
    /// (`BitextMining`, `Reranking`, …) or, failing that, a task description
    /// used as `task: {value} | query: `.
    Custom(String),
}

impl EmbeddingTask {
    /// Canonical names and aliases (case-insensitive, `-`/` ` treated as `_`),
    /// including the Gemini `taskType` enum values. Unknown text becomes
    /// [`EmbeddingTask::Custom`].
    pub fn parse(value: &str) -> Self {
        match normalize_key(value).as_str() {
            "document" | "retrieval_document" | "search_document" | "index" | "passage"
            | "generic_index" => Self::Document,
            "query" | "retrieval_query" | "search_query" | "search" | "generic_retrieval"
            | "text_retrieval" | "search_result" => Self::Query,
            "classification" | "classify" => Self::Classification,
            "clustering" | "cluster" => Self::Clustering,
            "similarity" | "semantic_similarity" | "sts" | "sentence_similarity" => {
                Self::Similarity
            }
            "code_query" | "code_retrieval_query" | "code" | "code_retrieval" => Self::CodeQuery,
            "fact_verification" | "fact" | "fact_checking" => Self::FactVerification,
            "question_answering" | "qa" => Self::QuestionAnswering,
            _ => Self::Custom(value.trim().to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Document => "document",
            Self::Query => "query",
            Self::Classification => "classification",
            Self::Clustering => "clustering",
            Self::Similarity => "similarity",
            Self::CodeQuery => "code_query",
            Self::FactVerification => "fact_verification",
            Self::QuestionAnswering => "question_answering",
            Self::Custom(raw) => raw,
        }
    }

    /// The canonical tasks, for `/v1/models`.
    pub const CANONICAL: [EmbeddingTask; 8] = [
        Self::Document,
        Self::Query,
        Self::Classification,
        Self::Clustering,
        Self::Similarity,
        Self::CodeQuery,
        Self::FactVerification,
        Self::QuestionAnswering,
    ];
}

impl Serialize for EmbeddingTask {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for EmbeddingTask {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::parse(&String::deserialize(deserializer)?))
    }
}

fn normalize_key(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

/// Prompt names compared without case or punctuation (`Retrieval-query` =
/// `retrieval_query` = `RetrievalQuery`).
fn loose_key(value: &str) -> String {
    value
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// EmbeddingGemma 2's task prefixes.
#[derive(Debug, Clone, PartialEq)]
pub struct Prompts {
    /// Name → prefix, as in `config_sentence_transformers.json`.
    pub named: BTreeMap<String, String>,
}

impl Default for Prompts {
    /// The prompts shipped with `google/embeddinggemma-2`.
    fn default() -> Self {
        let named = [
            ("query", "task: search result | query: "),
            ("document", "title: none | text: "),
            ("BitextMining", "task: search result | query: "),
            ("Classification", "task: classification | query: "),
            ("Clustering", "task: clustering | query: "),
            ("CodeRetrieval", "task: code retrieval | query: "),
            ("FactChecking", "task: fact checking | query: "),
            ("InstructionRetrieval", "task: code retrieval | query: "),
            ("MultilabelClassification", "task: classification | query: "),
            ("PairClassification", "task: sentence similarity | query: "),
            ("QuestionAnswering", "task: question answering | query: "),
            ("Reranking", "task: search result | query: "),
            ("Retrieval", "task: search result | query: "),
            ("Retrieval-document", "title: none | text: "),
            ("Retrieval-query", "task: search result | query: "),
            ("STS", "task: sentence similarity | query: "),
            ("SearchQuery", "task: search result | query: "),
            ("SentenceSimilarity", "task: sentence similarity | query: "),
            ("Summarization", "task: sentence similarity | query: "),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        Self { named }
    }
}

impl Prompts {
    /// The checkpoint's prompts over the built-in ones.
    pub fn with_overrides(named: BTreeMap<String, String>) -> Self {
        let mut p = Self::default();
        p.named.extend(named);
        p
    }

    /// Look a prompt up by name, ignoring case and punctuation.
    pub fn named(&self, name: &str) -> Option<&str> {
        let key = loose_key(name);
        self.named
            .iter()
            .find(|(k, _)| loose_key(k) == key)
            .map(|(_, v)| v.as_str())
    }

    /// The prefix for a task. Documents with a `title` use `title: {title} |
    /// text: ` (the model card's convention).
    pub fn prefix(&self, task: &EmbeddingTask, title: Option<&str>) -> String {
        let name = match task {
            EmbeddingTask::Document => {
                if let Some(t) = title.map(str::trim).filter(|t| !t.is_empty()) {
                    return format!("title: {t} | text: ");
                }
                "document"
            }
            EmbeddingTask::Query => "query",
            EmbeddingTask::Classification => "Classification",
            EmbeddingTask::Clustering => "Clustering",
            EmbeddingTask::Similarity => "STS",
            EmbeddingTask::CodeQuery => "CodeRetrieval",
            EmbeddingTask::FactVerification => "FactChecking",
            EmbeddingTask::QuestionAnswering => "QuestionAnswering",
            EmbeddingTask::Custom(raw) => {
                return self
                    .named(raw)
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("task: {} | query: ", raw.trim()));
            }
        };
        self.named(name).map(str::to_string).unwrap_or_else(|| {
            Prompts::default()
                .named(name)
                .unwrap_or_default()
                .to_string()
        })
    }
}

/// How text inputs are prefixed: an explicit `prompt` wins, else the task
/// (default `document`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PromptChoice {
    pub task: Option<EmbeddingTask>,
    pub prompt: Option<String>,
    pub title: Option<String>,
}

impl PromptChoice {
    pub fn prefix(&self, prompts: &Prompts) -> String {
        match &self.prompt {
            Some(p) => p.clone(),
            None => prompts.prefix(
                self.task.as_ref().unwrap_or(&EmbeddingTask::Document),
                self.title.as_deref(),
            ),
        }
    }
}

/// `POST /v1/embeddings` body — wire-compatible with `GaiseEmbeddingsRequest`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbeddingsRequest {
    /// `embeddinggemma::embeddinggemma-2`, `embeddinggemma-2`,
    /// `google/embeddinggemma-2`, or empty for the service default.
    #[serde(default)]
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    pub input: OneOrMany<String>,
    /// `document` (default), `query`, `classification`, `clustering`,
    /// `similarity`, `code_query`, `fact_verification`,
    /// `question_answering`, their aliases, a prompt name from the
    /// checkpoint (`BitextMining`, `Reranking`), or a task description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<EmbeddingTask>,
    /// Extension: a literal prefix to use instead of the task's (`""` for
    /// none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Extension: document title for `title: {title} | text: `.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Matryoshka size: snapped to 128, 256, 512 or 768 (the default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// Unset or `true` → unit-length (re-normalized after truncation);
    /// `false` → the raw mean-pooled vector (truncated, not re-normalized).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,
}

impl EmbeddingsRequest {
    pub fn prompt_choice(&self) -> PromptChoice {
        PromptChoice {
            task: self.task.clone(),
            prompt: self.prompt.clone(),
            title: self.title.clone(),
        }
    }
}

/// `POST /v1/embeddings/contents` body — wire-compatible with
/// `GaiseContentEmbeddingsRequest`: one vector per item; a `parts` item
/// yields one aggregated vector of all its parts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContentEmbeddingsRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    pub input: OneOrMany<Content>,
    /// Applied to text-only items; media items carry no prefix (the model
    /// card's guidance), unless `prompt` is given explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<EmbeddingTask>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,
}

impl ContentEmbeddingsRequest {
    pub fn prompt_choice(&self) -> PromptChoice {
        PromptChoice {
            task: self.task.clone(),
            prompt: self.prompt.clone(),
            title: self.title.clone(),
        }
    }
}

/// Token counters — wire-compatible with `GaiseUsage`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct Usage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<BTreeMap<String, usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<BTreeMap<String, usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<BTreeMap<String, usize>>,
}

/// Tokens fed to the model, by modality. Media counts are soft tokens; the
/// begin/end markers around them count as text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenCounts {
    pub text: usize,
    pub image: usize,
    pub video: usize,
    pub audio: usize,
}

impl TokenCounts {
    pub fn total(&self) -> usize {
        self.text + self.image + self.video + self.audio
    }

    pub fn add(&mut self, other: &TokenCounts) {
        self.text += other.text;
        self.image += other.image;
        self.video += other.video;
        self.audio += other.audio;
    }
}

impl Usage {
    /// `input.prompt_tokens` (+ `text_tokens`, `image_tokens`, … when media
    /// were embedded), `total.total_tokens`.
    pub fn from_counts(c: &TokenCounts) -> Self {
        let mut input = BTreeMap::from([("prompt_tokens".to_string(), c.total())]);
        if c.image + c.video + c.audio > 0 {
            for (k, v) in [
                ("text_tokens", c.text),
                ("image_tokens", c.image),
                ("video_tokens", c.video),
                ("audio_tokens", c.audio),
            ] {
                if v > 0 {
                    input.insert(k.to_string(), v);
                }
            }
        }
        Self {
            input: Some(input),
            output: None,
            total: Some(BTreeMap::from([("total_tokens".to_string(), c.total())])),
        }
    }
}

/// Response body — wire-compatible with `GaiseEmbeddingsResponse`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbeddingsResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,
    /// One vector per input, in input order.
    pub output: Vec<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Snap a requested size onto the Matryoshka set (nearest; ties go up, as
/// GAISe's `snap_dimensions`). `None` → the full size.
pub fn resolve_dimensions(requested: Option<u32>, supported: &[u32], full: u32) -> u32 {
    let Some(r) = requested else {
        return full;
    };
    supported
        .iter()
        .copied()
        .min_by_key(|d| (d.abs_diff(r), u32::from(*d < r)))
        .unwrap_or(full)
}

/// Truncate to `dims`, then L2-normalize when `normalize`.
pub fn finish_vector(mut v: Vec<f32>, dims: usize, normalize: bool) -> Vec<f32> {
    v.truncate(dims);
    if normalize {
        let norm = v
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x = (*x as f64 / norm) as f32;
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gaise_and_gemini_task_names() {
        assert_eq!(EmbeddingTask::parse("search_query"), EmbeddingTask::Query);
        assert_eq!(
            EmbeddingTask::parse("RETRIEVAL_QUERY"),
            EmbeddingTask::Query
        );
        assert_eq!(
            EmbeddingTask::parse("RETRIEVAL_DOCUMENT"),
            EmbeddingTask::Document
        );
        assert_eq!(
            EmbeddingTask::parse("SEMANTIC_SIMILARITY"),
            EmbeddingTask::Similarity
        );
        assert_eq!(
            EmbeddingTask::parse("CODE_RETRIEVAL_QUERY"),
            EmbeddingTask::CodeQuery
        );
        assert_eq!(EmbeddingTask::parse("qa"), EmbeddingTask::QuestionAnswering);
        assert_eq!(
            EmbeddingTask::parse("BitextMining"),
            EmbeddingTask::Custom("BitextMining".into())
        );
        let t: EmbeddingTask = serde_json::from_str("\"search-query\"").unwrap();
        assert_eq!(t, EmbeddingTask::Query);
        assert_eq!(
            serde_json::to_string(&EmbeddingTask::CodeQuery).unwrap(),
            "\"code_query\""
        );
    }

    #[test]
    fn prefixes_follow_the_checkpoint() {
        let p = Prompts::default();
        assert_eq!(
            p.prefix(&EmbeddingTask::Query, None),
            "task: search result | query: "
        );
        assert_eq!(
            p.prefix(&EmbeddingTask::Document, None),
            "title: none | text: "
        );
        assert_eq!(
            p.prefix(&EmbeddingTask::Document, Some("Proteins")),
            "title: Proteins | text: "
        );
        assert_eq!(
            p.prefix(&EmbeddingTask::Similarity, None),
            "task: sentence similarity | query: "
        );
        assert_eq!(
            p.prefix(&EmbeddingTask::Custom("bitext-mining".into()), None),
            "task: search result | query: "
        );
        assert_eq!(
            p.prefix(&EmbeddingTask::Custom("legal statute lookup".into()), None),
            "task: legal statute lookup | query: "
        );
        let choice = PromptChoice {
            prompt: Some(String::new()),
            ..Default::default()
        };
        assert_eq!(choice.prefix(&p), "");
        let custom = Prompts::with_overrides(BTreeMap::from([("query".into(), "Q: ".into())]));
        assert_eq!(custom.prefix(&EmbeddingTask::Query, None), "Q: ");
    }

    #[test]
    fn snaps_and_finishes() {
        let set = [768, 512, 256, 128];
        assert_eq!(resolve_dimensions(None, &set, 768), 768);
        assert_eq!(resolve_dimensions(Some(300), &set, 768), 256);
        assert_eq!(resolve_dimensions(Some(384), &set, 768), 512, "tie goes up");
        assert_eq!(resolve_dimensions(Some(5000), &set, 768), 768);
        assert_eq!(resolve_dimensions(Some(1), &set, 768), 128);
        assert_eq!(finish_vector(vec![3.0, 4.0, 12.0], 2, true), vec![0.6, 0.8]);
        assert_eq!(
            finish_vector(vec![3.0, 4.0, 12.0], 2, false),
            vec![3.0, 4.0]
        );
    }

    #[test]
    fn wire_shapes_match_gaise() {
        let r: EmbeddingsRequest =
            serde_json::from_str(r#"{"model":"x","input":["a","b"],"task":"search_query","dimensions":256,"connection":{"x":1}}"#)
                .unwrap();
        assert_eq!(r.input.len(), 2);
        assert_eq!(r.task, Some(EmbeddingTask::Query));
        let r: EmbeddingsRequest = serde_json::from_str(r#"{"input":"single"}"#).unwrap();
        assert_eq!(r.input, OneOrMany::One("single".into()));
        let resp = EmbeddingsResponse {
            external_id: None,
            output: vec![vec![0.5, 0.5]],
            usage: Some(Usage::from_counts(&TokenCounts {
                text: 3,
                ..Default::default()
            })),
        };
        assert_eq!(
            serde_json::to_string(&resp).unwrap(),
            r#"{"output":[[0.5,0.5]],"usage":{"input":{"prompt_tokens":3},"total":{"total_tokens":3}}}"#
        );
        let u = Usage::from_counts(&TokenCounts {
            text: 4,
            image: 280,
            ..Default::default()
        });
        assert_eq!(u.input.unwrap()["image_tokens"], 280);
    }
}
