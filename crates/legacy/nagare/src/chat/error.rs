use serde::{Deserialize, Serialize};

#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::util::power::Error as EnergyError;

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

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl From<EnergyError> for ChatSessionError {
    fn from(error: EnergyError) -> Self {
        Self::Backend {
            message: error.to_string(),
        }
    }
}
