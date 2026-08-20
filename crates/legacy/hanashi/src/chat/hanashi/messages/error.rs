use shoji::types::session::chat::{ChatContentBlockType, ChatRole};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Role '{role}' is not supported")]
    UnsupportedRole {
        role: ChatRole,
    },
    #[error("Serialization failed: {reason}")]
    SerializationFailed {
        reason: String,
    },
    #[error("Content block '{block_type}' is not supported for role '{role}'")]
    UnsupportedBlock {
        role: ChatRole,
        block_type: ChatContentBlockType,
    },
    #[error("Multiple '{block_type}' blocks are not allowed for role '{role}'")]
    MultipleNotAllowed {
        role: ChatRole,
        block_type: ChatContentBlockType,
    },
    #[error("Value of '{block_type}' is not allowed for role '{role}'")]
    ValueNotAllowed {
        role: ChatRole,
        block_type: ChatContentBlockType,
        allowed_values: String,
    },
    #[error("Cannot map '{value}' for '{block_type}' on role '{role}'")]
    UnmappedValue {
        role: ChatRole,
        block_type: ChatContentBlockType,
        value: String,
    },
    #[error("Field '{field}' is required for role '{role}' but was not provided")]
    FieldRequired {
        role: ChatRole,
        field: String,
    },
    #[error("Block type '{block_type}' is assigned to multiple fields for role '{role}'")]
    DuplicateBlock {
        role: ChatRole,
        block_type: ChatContentBlockType,
    },
    #[error("Limit of {limit} exceeded for '{block_type}' on role '{role}'")]
    LimitExceeded {
        role: ChatRole,
        block_type: ChatContentBlockType,
        limit: usize,
    },
}
