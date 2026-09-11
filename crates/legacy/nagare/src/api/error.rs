use reqwest::StatusCode;

use crate::api::IsTransient;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("HTTP {code}: {body}")]
    Http {
        code: StatusCode,
        body: String,
    },
    #[error("Timeout")]
    Timeout,
    #[error("Network: {0}")]
    Network(String),
    #[error("Decode: {0}")]
    Decode(String),
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        if error.is_timeout() {
            Error::Timeout
        } else {
            Error::Network(error.to_string())
        }
    }
}

impl IsTransient for Error {
    fn is_transient(&self) -> bool {
        match self {
            Self::Timeout | Self::Network(_) => true,
            Self::Http {
                code,
                ..
            } => code.is_transient(),
            Self::Decode(_) => false,
        }
    }
}
