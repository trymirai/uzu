use crate::locks::LockError;

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("downloaded file is {actual} bytes but registry declared {expected}")]
    Size {
        expected: u64,
        actual: u64,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Lock(#[from] LockError),
    #[cfg(target_vendor = "apple")]
    #[error(transparent)]
    Apple(#[from] crate::backends::AppleBackendError),
}
