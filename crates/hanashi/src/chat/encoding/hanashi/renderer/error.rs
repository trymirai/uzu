use crate::chat::encoding::hanashi::messages::Error as MessageError;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Invalid template")]
    InvalidTemplate,
    #[error("Render failed: {reason}")]
    RenderFailed {
        reason: String,
    },
    #[error("BOS token is required")]
    BosTokenRequired,
    #[error("EOS token is required")]
    EosTokenRequired,
    #[error("Duplicate context key: '{key}'")]
    DuplicateContextKey {
        key: String,
    },
    #[error(transparent)]
    Message(#[from] MessageError),
}
