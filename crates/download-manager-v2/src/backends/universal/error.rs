#[derive(thiserror::Error, Clone, Debug, PartialEq, Eq)]
pub enum UniversalBackendError {
    #[error("universal backend is not wired yet")]
    NotWired,
    #[error("io error: {0}")]
    Io(String),
}
