#[derive(thiserror::Error, Clone, Debug, PartialEq, Eq)]
pub enum AppleBackendError {
    #[error("bad url")]
    BadUrl,
    #[error("resume data error: {0}")]
    ResumeData(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("task enumeration error: {0}")]
    TaskEnumeration(String),
    #[error("authentication error: {0}")]
    Authentication(String),
}
