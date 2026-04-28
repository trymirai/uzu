#[derive(thiserror::Error, Debug)]
pub enum DownloadError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    SerdeJson(#[from] serde_json::Error),
    #[error("http client error: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("http error: status {0}")]
    HttpStatus(u16),
    #[error("canceled")]
    Canceled,
    #[error("resume unsupported")]
    ResumeUnsupported,
    #[error("io error: {0}")]
    IOError(String),
    #[error("unsupported file download manager type")]
    UnsupportedType,

    // Concrete cases instead of stringly-typed variants
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
}
