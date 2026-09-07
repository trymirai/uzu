#[derive(thiserror::Error, Clone, Debug, PartialEq, Eq)]
pub enum DownloadError {
    #[error("io error: {0}")]
    Io(String),
    #[error("json error: {0}")]
    SerdeJson(String),
    #[error("mutex poisoned")]
    MutexPoisoned,
    #[error("invalid state transition")]
    InvalidStateTransition,
    #[error("file locked by another manager: {0}")]
    LockedByOther(String),
    #[error("conflicting download config for destination: {0}")]
    ConflictingConfig(String),
    #[error("channel closed")]
    ChannelClosed,
    #[error("backend error: {0}")]
    Backend(String),
}

impl From<std::io::Error> for DownloadError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

impl From<serde_json::Error> for DownloadError {
    fn from(error: serde_json::Error) -> Self {
        Self::SerdeJson(error.to_string())
    }
}
