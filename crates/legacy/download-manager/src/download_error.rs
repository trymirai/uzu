use crate::{backends::BackendError, file_download::FileDownloadError, locks::LockError};

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error(transparent)]
    Lock(#[from] LockError),
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    FileDownload(#[from] FileDownloadError),
    #[error("conflicting download config for destination: {0}")]
    ConflictingConfig(String),
}
