use shoji::types::encoding::{ContentBlockType, Role};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Role '{role}' is not supported")]
    UnsupportedRole {
        role: Role,
    },
    #[error("Serialization failed: {reason}")]
    SerializationFailed {
        reason: String,
    },
    #[error("Content block '{block_type}' is not supported for role '{role}'")]
    UnsupportedBlock {
        role: Role,
        block_type: ContentBlockType,
    },
    #[error("Multiple '{block_type}' blocks are not allowed for role '{role}'")]
    MultipleNotAllowed {
        role: Role,
        block_type: ContentBlockType,
    },
    #[error("Value of '{block_type}' is not allowed for role '{role}'")]
    ValueNotAllowed {
        role: Role,
        block_type: ContentBlockType,
        allowed_values: String,
    },
    #[error("Cannot map '{value}' for '{block_type}' on role '{role}'")]
    UnmappedValue {
        role: Role,
        block_type: ContentBlockType,
        value: String,
    },
    #[error("Field '{field}' is required for role '{role}' but was not provided")]
    FieldRequired {
        role: Role,
        field: String,
    },
    #[error("Block type '{block_type}' is assigned to multiple fields for role '{role}'")]
    DuplicateBlock {
        role: Role,
        block_type: ContentBlockType,
    },
    #[error("Limit of {limit} exceeded for '{block_type}' on role '{role}'")]
    LimitExceeded {
        role: Role,
        block_type: ContentBlockType,
        limit: usize,
    },
}
