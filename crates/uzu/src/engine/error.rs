use nagare::chat::Error as ChatError;

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
    #[error("Backend error: {message}")]
    Backend {
        message: String,
    },
    #[error("Unsupported model")]
    UnsupportedModel,
    #[error(transparent)]
    Chat(#[from] ChatError),
}
