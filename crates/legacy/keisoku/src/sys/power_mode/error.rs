#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("native power mode APIs are unavailable")]
    Unavailable,
    #[error("High Power Mode is unsupported for {0} power")]
    Unsupported(&'static str),
    #[error("changing power mode requires root privileges")]
    RootRequired,
    #[error("power mode {operation} failed with status {code:#x}")]
    PlatformCall {
        operation: &'static str,
        code: i32,
    },
    #[error("power mode preference is missing or invalid")]
    InvalidPreference,
    #[error("power mode readback was {actual} after requesting {requested}")]
    ReadbackMismatch {
        requested: i64,
        actual: i64,
    },
    #[error("power mode change failed: {operation}; restoration also failed: {restore}")]
    RollbackFailed {
        #[source]
        operation: Box<Self>,
        restore: Box<Self>,
    },
}
