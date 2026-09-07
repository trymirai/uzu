#[derive(Debug, thiserror::Error)]
pub enum KeisokuError {
    #[cfg(all(target_os = "macos", feature = "hardware-control"))]
    #[error(transparent)]
    GpuControl(#[from] crate::GpuControlError),
    #[cfg(all(target_os = "macos", feature = "hardware-control"))]
    #[error(transparent)]
    FanControl(#[from] crate::SmcError),
    #[cfg(all(target_os = "macos", feature = "hardware-control"))]
    #[error(transparent)]
    PowerMode(#[from] crate::PowerModeError),
    #[error("power meter has not been started")]
    PowerMeterNotStarted,
    #[error("power meter did not produce a reading")]
    PowerReadingUnavailable,
    #[error("power meter sampling thread disconnected")]
    SamplingTaskDisconnected,
    #[error("power meter sampling thread panicked")]
    SamplingTaskPanicked,
}
