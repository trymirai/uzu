use std::io::Error as IoError;

use serde_json::Error as JsonError;

#[derive(thiserror::Error, Clone, Debug, PartialEq, Eq)]
pub enum DownloadError {
    #[error("io error: {0}")]
    Io(String),
    #[error("json error: {0}")]
    SerdeJson(String),
    #[error("http error: status {0}")]
    HttpStatus(u16),
    #[error("canceled")]
    Canceled,
    #[error("resume unsupported")]
    ResumeUnsupported,
    #[error("bad url")]
    BadUrl,
    #[error("authenticated downloads require HTTPS")]
    InsecureAuthenticatedRequest,
    #[error("download authentication is no longer available")]
    AuthenticationUnavailable,
    #[error("missing download info for task")]
    MissingDownloadInfo,
    #[error("resume data read failed")]
    ResumeDataReadFailed,
    #[error("resume data parsing or handling error")]
    ResumeDataError,
    #[error("download task not found")]
    DownloadTaskNotFound,
    #[error("task not found after creation")]
    TaskNotFoundAfterCreation,
    #[error("no matching download task to pause")]
    NoMatchingTaskToPause,
    #[error("unknown download handle")]
    UnknownDownloadHandle,
    #[error("mutex poisoned")]
    MutexPoisoned,
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
}

impl From<IoError> for DownloadError {
    fn from(error: IoError) -> Self {
        Self::Io(error.to_string())
    }
}

impl From<JsonError> for DownloadError {
    fn from(error: JsonError) -> Self {
        Self::SerdeJson(error.to_string())
    }
}
