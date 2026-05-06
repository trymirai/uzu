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
    #[error("unsupported file download manager type")]
    UnsupportedType,
    #[error("bad url")]
    BadUrl,
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
    #[error("task stopped")]
    TaskStopped,
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
