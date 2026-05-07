#[derive(thiserror::Error, Clone, Debug, PartialEq, Eq)]
pub enum UniversalBackendError {
    #[error("io error: {0}")]
    Io(String),
}
