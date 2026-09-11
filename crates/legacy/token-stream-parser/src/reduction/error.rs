use crate::types::TokenValue;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ReductionParserError {
    #[error("Invalid state: expected group at index {index}")]
    InvalidState {
        index: usize,
    },
    #[error("Duplicate group name at same level: {name}")]
    DuplicateGroupName {
        name: String,
    },
    #[error("Duplicate open token at same level: {token}")]
    DuplicateOpenToken {
        token: TokenValue,
    },
}
