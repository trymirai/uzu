use crate::locks::LockError;

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Lock(#[from] LockError),
    #[cfg(target_vendor = "apple")]
    #[error(transparent)]
    Apple(#[from] crate::backends::AppleBackendError),
}
