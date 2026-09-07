#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("GPU power control is unavailable")]
    Unavailable,
    #[error("GPU calibrated maximum power could not be read")]
    MaximumPowerUnavailable,
    #[error("GPU {operation} failed with driver status {code:#x}")]
    DriverCall {
        operation: &'static str,
        code: i32,
    },
}
