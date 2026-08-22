#[derive(thiserror::Error, Clone, Debug, PartialEq, Eq)]
pub enum UniversalBackendError {
    #[error(transparent)]
    RecoveryMetadata(#[from] crate::DownloadError),
    #[error("io error: {0}")]
    Io(String),
}
