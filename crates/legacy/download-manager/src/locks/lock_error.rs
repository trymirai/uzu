#[derive(Debug, thiserror::Error)]
pub enum LockError {
    #[error("destination locked by another manager: {manager_id}")]
    LockedByOther {
        manager_id: String,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
