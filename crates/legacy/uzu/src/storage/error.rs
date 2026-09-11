use download_manager::DownloadError;
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
    #[error("Hash not found for file: {identifier}/{name}")]
    HashNotFound {
        identifier: ModelIdentifier,
        name: String,
    },
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
        Self::DownloadManager {
            message: error.to_string(),
        }
    }
}
