#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("HTTP {code}: {body}")]
    Http {
        code: u16,
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
