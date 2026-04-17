#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Tokio error: {message}")]
    TokioError {
        message: String,
    },
    #[error(transparent)]
    Device(#[from] nagare::device::Error),
    #[error(transparent)]
    Storage(#[from] nagare::storage::Error),
    #[error(transparent)]
    Registry(#[from] nagare::registry::Error),
}
