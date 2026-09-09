#[derive(Debug, thiserror::Error)]
pub enum UniversalBackendError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error("{0}")]
    Protocol(String),
}
