use nagare::{chat::Error as ChatError, classification::Error as ClassificationError};

#[bindings::export(Error, name = "EngineError")]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Tokio error: {message}")]
    TokioError {
        message: String,
    },
    #[error(transparent)]
    Device(#[from] crate::device::Error),
    #[error(transparent)]
    Storage(#[from] crate::storage::Error),
    #[error(transparent)]
    Registry(#[from] crate::registry::Error),
    #[error("Unable to create backend")]
    UnableToCreateBackend,
    #[error("Backend not found")]
    BackendNotFound,
    #[error(transparent)]
    Chat(#[from] ChatError),
    #[error(transparent)]
    Classification(#[from] ClassificationError),
}
