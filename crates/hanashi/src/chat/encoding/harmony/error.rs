use crate::chat::encoding::{SynchronizationError, harmony::bridging::Error as BridgingError};
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Unable to load encoding")]
    UnableToLoadEncoding,
    #[error("Expected directory tokenizer location")]
    ExpectedDirectoryTokenizerLocation,
    #[error(transparent)]
    Bridging(#[from] BridgingError),
    #[error("Unable to render conversation")]
    UnableToRenderConversation,
    #[error("Unable to decode token")]
    UnableToDecodeToken,
    #[error(transparent)]
    Synchronization(#[from] SynchronizationError),
    #[error("Parser error: {reason}")]
    ParserError {
        reason: String,
    },
}
