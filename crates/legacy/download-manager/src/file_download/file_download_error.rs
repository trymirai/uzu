use crate::{backends::BackendError, locks::LockError};

#[derive(Debug, thiserror::Error)]
pub enum FileDownloadError {
    #[error("channel closed")]
    ChannelClosed,
    #[error(transparent)]
    Lock(#[from] LockError),
    #[error(transparent)]
    Backend(#[from] BackendError),
}
