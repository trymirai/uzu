use std::sync::PoisonError;

#[derive(Debug, thiserror::Error)]
pub enum KeisokuError {
    #[error("power meter has not been started")]
    PowerMeterNotStarted,
    #[error("power meter did not produce a reading")]
    PowerReadingUnavailable,
    #[error("power meter accumulator lock is poisoned")]
    AccumulatorPoisoned,
    #[error("power meter sampling thread panicked")]
    SamplingTaskPanicked,
}

impl<T> From<PoisonError<T>> for KeisokuError {
    fn from(_: PoisonError<T>) -> Self {
        Self::AccumulatorPoisoned
    }
}
