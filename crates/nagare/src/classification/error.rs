#[bindings::export(Error, name = "ClassificationSessionError")]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Backend error: {message}")]
    Backend {
        message: String,
    },
    #[error("Unsupported model")]
    UnsupportedModel,
    #[error("Unable to perform operation in current state")]
    UnableToPerformOperationInCurrentState,
    #[error("No response")]
    NoResponse,
}
