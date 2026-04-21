use shoji::types::session::chat::Role;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Expected one of [{expected}] after '{after}', got '{got}'")]
    InvalidTransition {
        after: String,
        expected: String,
        got: Role,
    },

    #[error("Expected one of [{expected}] at start, got '{got}'")]
    InvalidInitial {
        expected: String,
        got: Role,
    },

    #[error("No transitions defined for role '{role}'")]
    NoTransitions {
        role: Role,
    },
}
