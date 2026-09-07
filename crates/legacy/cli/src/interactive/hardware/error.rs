#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("The hardware control session has ended")]
    Closed,
    #[error(transparent)]
    PowerMode(#[from] keisoku::PowerModeError),
    #[error(transparent)]
    Fan(#[from] keisoku::SmcError),
    #[error("Hardware control I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("Administrator authorization failed: {0}")]
    Authorization(String),
    #[error("Hardware helper: {0}")]
    Protocol(&'static str),
    #[error("Hardware helper message is invalid: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Hardware helper task failed: {0}")]
    Worker(#[from] tokio::task::JoinError),
    #[error("{0}")]
    Remote(String),
    #[error("Another CLI session owns hardware control: {0}")]
    Lock(#[from] std::fs::TryLockError),
    #[error("{operation}; restoring the previous settings also failed: {restore}")]
    Rollback {
        #[source]
        operation: Box<Error>,
        restore: Box<Error>,
    },
}
