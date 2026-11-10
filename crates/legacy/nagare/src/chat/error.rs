use serde::{Deserialize, Serialize};
use shoji::{traits::backend::Error as ShojiBackendError, types::basic::BackendError};

#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::util::power::Error as EnergyError;

#[bindings::export(Error)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[non_exhaustive]
pub enum ChatSessionError {
    #[error("Backend error: {error}")]
    Backend {
        error: BackendError,
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

impl ChatSessionError {
    pub fn backend(message: impl ToString) -> Self {
        Self::Backend {
            error: BackendError::other(message),
        }
    }
}

impl From<ShojiBackendError> for ChatSessionError {
    fn from(error: ShojiBackendError) -> Self {
        Self::Backend {
            error: error.into(),
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl From<EnergyError> for ChatSessionError {
    fn from(error: EnergyError) -> Self {
        Self::backend(error)
    }
}
