use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("needle library not found (tried {tried:?}); set NEEDLE3_LIB_PATH")]
    LibraryNotFound {
        tried: Vec<PathBuf>,
    },
    #[error("failed to load needle library: {message}")]
    LibraryLoad {
        message: String,
    },
    #[error("missing C symbol {symbol}")]
    MissingSymbol {
        symbol: String,
    },
    #[error("no needle3 .cact found (tried {tried:?}); set NEEDLE_WEIGHTS_PATH")]
    WeightsNotFound {
        tried: Vec<PathBuf>,
    },
    #[error("{path} is not a Needle 3 archive (tag 0x{tag:08x})")]
    UnsupportedGeneration {
        path: PathBuf,
        tag: u32,
    },
    #[error("failed to read {path}: {message}")]
    ReadFailed {
        path: PathBuf,
        message: String,
    },
    #[error("needle_load failed for {path}")]
    LoadFailed {
        path: PathBuf,
    },
    #[error("needle already loaded {loaded}, cannot switch to {requested}")]
    WeightsLocked {
        loaded: PathBuf,
        requested: PathBuf,
    },
    #[error("needle_init failed")]
    InitFailed,
    #[error("needle_complete failed (code {code}): {message}")]
    CompleteFailed {
        code: i32,
        message: String,
    },
    #[error("engine returned invalid JSON: {message}")]
    InvalidEnvelope {
        message: String,
    },
    #[error("duplicate tool name `{name}` across namespaces")]
    DuplicateToolName {
        name: String,
    },
    #[error("unsupported content in chat messages")]
    UnsupportedContent,
    #[error("regex grammar is not supported by Needle")]
    UnsupportedGrammar,
    #[error("cancelled")]
    Cancelled,
    #[error("needle session state is invalid")]
    InvalidState,
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::InvalidEnvelope {
            message: error.to_string(),
        }
    }
}
