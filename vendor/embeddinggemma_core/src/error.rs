use std::fmt;

/// Everything that can go wrong in `embeddinggemma_core`.
#[derive(Debug)]
pub enum EmbedError {
    /// A tensor / device / weight-loading failure from candle.
    Candle(candle_core::Error),
    /// A failure from a non-candle backend (MLX).
    Backend(String),
    /// Tokenizer load or encode failure.
    Tokenizer(String),
    /// File system failure while locating model files.
    Io(std::io::Error),
    /// Hugging Face Hub download failure.
    Hub(String),
    /// `config.json` could not be parsed or describes an unsupported model.
    Config(String),
    /// An image / audio / video / PDF could not be decoded.
    Media(String),
    /// The caller's request was invalid (empty input, unknown model, …).
    InvalidRequest(String),
    /// The requested device or dtype is not available in this build.
    Device(String),
}

impl fmt::Display for EmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Candle(e) => write!(f, "inference error: {e}"),
            Self::Backend(e) => write!(f, "inference error: {e}"),
            Self::Tokenizer(e) => write!(f, "tokenizer error: {e}"),
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Hub(e) => write!(f, "hub error: {e}"),
            Self::Config(e) => write!(f, "model config error: {e}"),
            Self::Media(e) => write!(f, "media error: {e}"),
            Self::InvalidRequest(e) => write!(f, "invalid request: {e}"),
            Self::Device(e) => write!(f, "device error: {e}"),
        }
    }
}

impl std::error::Error for EmbedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Candle(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl EmbedError {
    /// Errors the caller can fix (bad input, undecodable media) as opposed to
    /// server-side failures.
    pub fn is_client_error(&self) -> bool {
        matches!(self, Self::InvalidRequest(_) | Self::Media(_))
    }
}

impl From<candle_core::Error> for EmbedError {
    fn from(e: candle_core::Error) -> Self {
        Self::Candle(e)
    }
}

impl From<std::io::Error> for EmbedError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<tokenizers::Error> for EmbedError {
    fn from(e: tokenizers::Error) -> Self {
        Self::Tokenizer(e.to_string())
    }
}

impl From<serde_json::Error> for EmbedError {
    fn from(e: serde_json::Error) -> Self {
        Self::Config(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, EmbedError>;
