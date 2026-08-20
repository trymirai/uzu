#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unable to create: {message}")]
    UnableToCreate {
        message: String,
    },
    #[error("Network error: {message}")]
    Network {
        message: String,
    },
    #[error("Serialization error: {message}")]
    Serialization {
        message: String,
    },
    #[error("Unsupported role")]
    UnsupportedRole,
    #[error("Tool call result is required")]
    ToolCallResultRequired,
    #[error("Content is required")]
    ContentRequired,
}
