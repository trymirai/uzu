use shoji::types::session::chat::ChatRole;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Expected one of [{expected}] after '{after}', got '{got}'")]
    InvalidTransition {
        after: String,
        expected: String,
        got: ChatRole,
    },

    #[error("Expected one of [{expected}] at start, got '{got}'")]
    InvalidInitial {
        expected: String,
        got: ChatRole,
    },

    #[error("No transitions defined for role '{role}'")]
    NoTransitions {
        role: ChatRole,
    },
}
