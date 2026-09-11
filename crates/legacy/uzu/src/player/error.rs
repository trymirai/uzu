#[bindings::export(Error)]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PlayerError {
    #[error("Rodio error: {message}")]
    RodioError {
        message: String,
    },
    #[error("Invalid PCM batch: {message}")]
    InvalidPcmBatch {
        message: String,
    },
}
