use serde::{Deserialize, Serialize};

#[bindings::export(Error)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[non_exhaustive]
pub enum ChatSessionError {
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
