#[derive(Debug, thiserror::Error)]
pub enum KeisokuError {
    #[error("power meter has not been started")]
    PowerMeterNotStarted,
    #[error("power meter did not produce a reading")]
    PowerReadingUnavailable,
    #[error("power meter sampling thread disconnected")]
    SamplingTaskDisconnected,
    #[error("power meter sampling thread panicked")]
    SamplingTaskPanicked,
}
