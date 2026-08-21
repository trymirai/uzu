#[bindings::export(Error)]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ClassificationSessionError {
    #[error("Backend error: {message}")]
    Backend {
        message: String,
    },
    #[error("Unsupported model")]
    UnsupportedModel {},
    #[error("Unable to perform operation in current state")]
    UnableToPerformOperationInCurrentState {},
    #[error("No response")]
    NoResponse {},
}
