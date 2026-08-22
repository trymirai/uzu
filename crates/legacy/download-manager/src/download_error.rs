use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DownloadCleanupFailure {
    pub path: String,
    pub message: String,
}

impl DownloadCleanupFailure {
    pub(crate) fn new(
        path: &std::path::Path,
        error: &std::io::Error,
    ) -> Self {
        Self {
            path: path.display().to_string(),
            message: error.to_string(),
        }
    }
}

#[derive(thiserror::Error, Clone, Debug, PartialEq, Eq)]
pub enum DownloadError {
    #[error("io error: {0}")]
    Io(String),
    #[error("json error: {0}")]
    SerdeJson(String),
    #[error("http error: status {0}")]
    HttpStatus(u16),
    #[error("authentication required")]
    AuthenticationRequired,
    #[error("access denied")]
    AccessDenied,
    #[error("download source not found")]
    SourceNotFound,
    #[error("download source is gone")]
    SourceGone,
    #[error("transport error: {0}")]
    Transport(String),
    #[error("download protocol error: {0}")]
    Protocol(String),
    #[error("canceled")]
    Canceled,
    #[error("resume unsupported")]
    ResumeUnsupported,
    #[error("bad url")]
    BadUrl,
    #[error("invalid request header")]
    InvalidRequestHeader,
    #[error("authenticated downloads require HTTPS")]
    InsecureAuthenticatedRequest,
    #[error("invalid state transition")]
    InvalidStateTransition,
    #[error("file locked by another manager: {0}")]
    LockedByOther(String),
    #[error("conflicting download config for destination: {0}")]
    ConflictingConfig(String),
    #[error("task stopped")]
    TaskStopped,
    #[error("channel closed")]
    ChannelClosed,
    #[error("backend error: {0}")]
    Backend(String),
    #[error("invalid expected {algorithm} digest")]
    InvalidDigest {
        algorithm: &'static str,
    },
    #[error("integrity verification I/O failed for {path}: {message}")]
    IntegrityIo {
        path: String,
        message: String,
    },
    #[error("integrity mismatch: {0}")]
    IntegrityMismatch(String),
    #[error("cleanup failed for {} owned path(s)", .failures.len())]
    CleanupFailures {
        failures: Arc<[DownloadCleanupFailure]>,
    },
    #[error("legacy file download task does not support destructive cleanup")]
    DestructiveCleanupUnsupported,
}

impl DownloadError {
    pub(crate) fn from_http_status(status: u16) -> Self {
        match status {
            401 => Self::AuthenticationRequired,
            403 => Self::AccessDenied,
            404 => Self::SourceNotFound,
            410 => Self::SourceGone,
            status => Self::HttpStatus(status),
        }
    }

    pub(crate) fn cleanup_failures(failures: Vec<DownloadCleanupFailure>) -> Self {
        debug_assert!(!failures.is_empty());
        Self::CleanupFailures {
            failures: failures.into(),
        }
    }

    pub(crate) fn is_retryable_transfer_failure(&self) -> bool {
        matches!(self, Self::Transport(_))
            || matches!(self, Self::HttpStatus(status) if *status == 429 || (500..=599).contains(status))
    }
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

#[cfg(test)]
#[path = "../tests/unit/download_error_test.rs"]
mod tests;
