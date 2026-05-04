#[derive(thiserror::Error, Clone, Debug, PartialEq, Eq)]
pub enum AppleBackendError {
    #[error("apple backend is not wired yet")]
    NotWired,
    #[error("bad url")]
    BadUrl,
    #[error("resume data error: {0}")]
    ResumeData(String),
    #[error("io error: {0}")]
    Io(String),
}
