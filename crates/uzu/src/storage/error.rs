use crate::{registry::Error as RegistryError, storage::types::DownloadPhase};

#[bindings::export(Error, name = "StorageError")]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Unable to create directory: {path}")]
    UnableToCreateDirectory {
        path: String,
    },
    #[error("Unable to create download manager: {message}")]
    UnableToCreateDownloadManager {
        message: String,
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
    #[error("Invalid state transition from {from:?} to {to:?}")]
    InvalidStateTransition {
        from: DownloadPhase,
        to: DownloadPhase,
    },
    #[error("IO error: {message}")]
    IO {
        message: String,
    },
    #[error("Item not found: {identifier}")]
    ItemNotFound {
        identifier: String,
    },
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error("Unsupported item: {identifier}")]
    UnsupportedItem {
        identifier: String,
    },
}
