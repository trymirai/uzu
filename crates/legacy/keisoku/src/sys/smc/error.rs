#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("SMC service is unavailable")]
    Unavailable,
    #[error("SMC IOKit operation failed: {0:#x}")]
    IoKit(i32),
    #[error("SMC key {key:#x} returned firmware error {result:#x}")]
    Firmware {
        key: u32,
        result: u8,
    },
    #[error("SMC key {0:#x} is absent")]
    MissingKey(u32),
    #[error("invalid SMC data for key {key:#x}: {reason}")]
    InvalidData {
        key: u32,
        reason: &'static str,
    },
    #[cfg(feature = "hardware-control")]
    #[error("fan control requires root privileges")]
    RootRequired,
    #[cfg(feature = "hardware-control")]
    #[error("this machine has no controllable fans")]
    NoFans,
    #[cfg(feature = "hardware-control")]
    #[error("SMC key {0:#x} did not retain the requested value")]
    ReadbackMismatch(u32),
    #[cfg(feature = "hardware-control")]
    #[error("fan control failed: {operation}; restoration also failed: {restore}")]
    RollbackFailed {
        #[source]
        operation: Box<Error>,
        restore: Box<Error>,
    },
}
