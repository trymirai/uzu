#[derive(thiserror::Error, Clone, Debug, PartialEq, Eq)]
pub enum AppleBackendError {
    #[error(transparent)]
    RecoveryMetadata(#[from] crate::DownloadError),
    #[error("bad url")]
    BadUrl,
    #[error("invalid request header")]
    InvalidRequestHeader,
    #[error("resume data error: {0}")]
    ResumeData(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("task enumeration error: {0}")]
    TaskEnumeration(String),
}
