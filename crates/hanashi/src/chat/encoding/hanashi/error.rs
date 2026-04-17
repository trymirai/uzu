use token_stream_parser::token_stream::TokenStreamParserError;

use crate::chat::encoding::{
    SynchronizationError,
    hanashi::{messages::Error as MessageError, ordering::Error as OrderingError, renderer::Error as RendererError},
};
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Config '{0}' not found")]
    ConfigNotFound(String),
    #[error("Config '{0}' is invalid")]
    InvalidConfig(String),
    #[error("Unable to load tokenizer")]
    UnableToLoadTokenizer,
    #[error("Unable to encode text")]
    UnableToEncodeText,
    #[error("Unable to decode token")]
    UnableToDecodeToken,
    #[error(transparent)]
    Synchronization(#[from] SynchronizationError),
    #[error("Failed to parse streamed content")]
    InvalidStreamedContent,
    #[error(transparent)]
    Parsing(#[from] TokenStreamParserError),
    #[error(transparent)]
    Rendering(#[from] RendererError),
    #[error(transparent)]
    Ordering(#[from] OrderingError),
    #[error(transparent)]
    Message(#[from] MessageError),
}
