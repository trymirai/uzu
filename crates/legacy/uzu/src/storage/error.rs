use download_manager::{BackendError, DownloadError, FileDownloadError, LockError};
use shoji::types::model::ModelIdentifier;

#[bindings::export(Error)]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum StorageError {
    #[error("Unable to create directory: {path}")]
    UnableToCreateDirectory {
        path: String,
    },
    #[error("Download manager error: {message}")]
    DownloadManager {
        message: String,
    },
    #[error("Model is locked by another manager")]
    Locked {},
    #[error("Model not found: {identifier}")]
    ModelNotFound {
        identifier: ModelIdentifier,
    },
    #[error("Unsupported model: {identifier}")]
    UnsupportedModel {
        identifier: ModelIdentifier,
    },
}

impl From<DownloadError> for StorageError {
    fn from(error: DownloadError) -> Self {
        match error {
            DownloadError::Lock(LockError::LockedByOther {
                ..
            })
            | DownloadError::Backend(BackendError::Lock(LockError::LockedByOther {
                ..
            }))
            | DownloadError::FileDownload(FileDownloadError::Lock(LockError::LockedByOther {
                ..
            }))
            | DownloadError::FileDownload(FileDownloadError::Backend(BackendError::Lock(LockError::LockedByOther {
                ..
            }))) => Self::Locked {},
            error => Self::DownloadManager {
                message: error.to_string(),
            },
        }
    }
}
