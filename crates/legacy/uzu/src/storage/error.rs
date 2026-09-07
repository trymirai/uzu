use download_manager::DownloadError;

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
        identifier: String,
        name: String,
    },
    #[error("Item not found: {identifier}")]
    ItemNotFound {
        identifier: String,
    },
    #[error("Unsupported item: {identifier}")]
    UnsupportedItem {
        identifier: String,
    },
}

impl From<DownloadError> for StorageError {
    fn from(error: DownloadError) -> Self {
        Self::DownloadManager {
            message: error.to_string(),
        }
    }
}
