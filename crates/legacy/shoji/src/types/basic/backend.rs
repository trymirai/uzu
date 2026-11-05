use serde::{Deserialize, Serialize};

use crate::traits::backend::Error as ShojiBackendError;

/// A backend failure that keeps its kind across the bindings boundary.
/// The variants mirror uzu-engine's LanguageModelStreamError so callers can
/// match on them (a context overflow, say) instead of on the error text.
#[bindings::export(Error)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[non_exhaustive]
pub enum BackendError {
    #[error("Backend error: {message}")]
    Backend {
        message: String,
    },
    #[error("Capture error: {message}")]
    Capture {
        message: String,
    },
    #[error("Decoder error: {message}")]
    Decoder {
        message: String,
    },
    #[error("Grammar error: {message}")]
    Grammar {
        message: String,
    },
    #[error("Trie accept error: {message}")]
    TrieAccept {
        message: String,
    },
    #[error("Speculator error: {message}")]
    Speculator {
        message: String,
    },
    #[error("No seed token (both state and input are empty)")]
    NoSeedToken {},
    #[error("Context overflow")]
    ContextOverflow {},
    /// Any failure that does not come from the language model stream.
    #[error("{message}")]
    Other {
        message: String,
    },
}

impl BackendError {
    pub fn other(message: impl ToString) -> Self {
        Self::Other {
            message: message.to_string(),
        }
    }
}

impl From<ShojiBackendError> for BackendError {
    fn from(error: ShojiBackendError) -> Self {
        match error.downcast::<BackendError>() {
            Ok(error) => *error,
            Err(error) => Self::other(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxed_backend_error_keeps_its_variant() {
        let boxed: ShojiBackendError = BackendError::ContextOverflow {}.into();
        assert_eq!(BackendError::from(boxed), BackendError::ContextOverflow {});
    }

    #[test]
    fn other_boxed_errors_become_other() {
        let boxed: ShojiBackendError = "Context overflow".into();
        assert_eq!(BackendError::from(boxed), BackendError::other("Context overflow"));
    }
}
