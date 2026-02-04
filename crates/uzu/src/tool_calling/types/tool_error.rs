use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Error, PartialEq, Serialize, Deserialize)]
pub enum ToolError {
    #[error("Tool not found: {name}")]
    NotFound { name: String },

    #[error("Invalid parameters: {message}")]
    InvalidParameters { message: String },

    #[error("Serialization error: {message}")]
    SerializationError { message: String },
}
