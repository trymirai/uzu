use serde::{Deserialize, Serialize};

use crate::util::power::Error as PowerError;

#[bindings::export(Error)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[non_exhaustive]
pub enum ChatSessionError {
    #[error("Backend error: {message}")]
    Backend {
        message: String,
    },
    #[error("Unable to load model: {message}")]
    Loading {
        message: String,
    },
    #[error("Unsupported model")]
    UnsupportedModel {},
    #[error("Unable to perform operation in current state")]
    UnableToPerformOperationInCurrentState {},
    #[error("No response")]
    NoResponse {},
    #[error("Tool turn limit exceeded: {limit}")]
    ToolTurnLimitExceeded {
        limit: u32,
    },
}

impl From<PowerError> for ChatSessionError {
    fn from(error: PowerError) -> Self {
        Self::Backend {
            message: error.to_string(),
        }
    }
}
