use tokio::sync::oneshot::error::RecvError;

#[derive(Debug, thiserror::Error)]
pub enum AppleBackendError {
    #[error("invalid url: {0}")]
    InvalidUrl(String),
    #[error("URLSession callback dropped: {0}")]
    CallbackDropped(#[from] RecvError),
}
