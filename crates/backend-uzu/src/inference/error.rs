use crate::session::types::Error as ChatSessionError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Runtime error: {message}")]
    Runtime {
        message: String,
    },
    #[error("Session error: {message}")]
    Session {
        message: String,
    },
    #[error("Unable to load tokenizer")]
    UnableToLoadTokenizer,
}

impl From<ChatSessionError> for Error {
    fn from(error: ChatSessionError) -> Self {
        Error::Session {
            message: error.to_string(),
        }
    }
}
