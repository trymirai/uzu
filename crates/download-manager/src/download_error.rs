#[derive(thiserror::Error, Debug)]
pub enum DownloadError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("io error: {0}")]
    IOError(String),
    #[error("unsupported file download manager type")]
    UnsupportedType,

    // Concrete cases instead of stringly-typed variants
    #[error("bad url")]
    BadUrl,
    #[error("resume data parsing or handling error")]
    ResumeDataError,
    #[error("task not found after creation")]
    TaskNotFoundAfterCreation,
    #[error("invalid state transition")]
    InvalidStateTransition,
    #[error("file locked by another manager: {0}")]
    LockedByOther(String),
}
