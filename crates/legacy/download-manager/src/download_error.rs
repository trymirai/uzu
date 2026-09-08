use crate::locks::LockError;

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("file locked by another manager: {0}")]
    LockedByOther(String),
    #[error("conflicting download config for destination: {0}")]
    ConflictingConfig(String),
    #[error("channel closed")]
    ChannelClosed,
    #[error("backend error: {0}")]
    Backend(String),
}

impl From<LockError> for DownloadError {
    fn from(error: LockError) -> Self {
        match error {
            LockError::LockedByOther {
                manager_id,
            } => Self::LockedByOther(manager_id),
            LockError::Io(error) => Self::Io(error),
        }
    }
}
