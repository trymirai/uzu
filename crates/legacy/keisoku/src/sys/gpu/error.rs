#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("GPU power control is unavailable")]
    Unavailable,
    #[error("GPU power limit could not be read")]
    PowerLimitUnavailable,
    #[error("GPU calibrated maximum power could not be read")]
    MaximumPowerUnavailable,
    #[error("GPU power limit must be positive")]
    InvalidPowerLimit,
    #[error("GPU power limit is outside the supported integer range")]
    PowerLimitOutOfRange(#[from] std::num::TryFromIntError),
    #[error("GPU {operation} failed with driver status {code:#x}")]
    DriverCall {
        operation: &'static str,
        code: i32,
    },
    #[error("GPU power limit readback was {actual} mW after requesting {requested} mW")]
    ReadbackMismatch {
        requested: u32,
        actual: u32,
    },
    #[error("GPU power change failed: {operation}; rollback also failed: {rollback}")]
    RollbackFailed {
        #[source]
        operation: Box<Self>,
        rollback: Box<Self>,
    },
}
