mod encoding_name;
mod message;
mod reasoning_effort;
mod role;
mod tool_namespace;

pub use message::{bridge_messages_from_harmony, bridge_messages_to_harmony};

use crate::chat::types::{ContentBlockType, ReasoningEffort, Role};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Role '{role}' is not supported")]
    UnsupportedRole {
        role: Role,
    },
    #[error("Reasoning effort '{reasoning_effort}' is not supported")]
    UnsupportedReasoningEffort {
        reasoning_effort: ReasoningEffort,
    },
    #[error("Builtin tool '{name}' is not supported")]
    UnsupportedBuiltinTool {
        name: String,
    },
    #[error("Content block '{block_type}' is not supported for role '{role}'")]
    UnsupportedContentBlock {
        block_type: ContentBlockType,
        role: Role,
    },
    #[error("Multiple tool calls are not supported")]
    MultipleToolCalls,
    #[error("Multiple content blocks are not supported")]
    MultipleContentBlocks,
    #[error("Content is required for role '{role}'")]
    ContentRequired {
        role: Role,
    },
    #[error("Tool call result is missing a name")]
    MissingToolCallResultName,
    #[error("Serialization failed: {message}")]
    SerializationFailed {
        message: String,
    },
}
