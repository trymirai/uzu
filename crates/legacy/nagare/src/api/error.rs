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
    /// The client has already retried by the time a caller sees this, so use it
    /// to decide about falling back to cached data, not about retrying.
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

#[cfg(test)]
mod tests {
    use reqwest::StatusCode;

    use super::Error;
    use crate::api::IsTransient;

    fn http(code: StatusCode) -> Error {
        Error::Http {
            code,
            body: String::new(),
        }
    }

    /// The status list itself is covered on `StatusCode`; this pins the
    /// delegation and the variants that carry no status.
    #[test]
    fn variants_are_classified() {
        assert!(http(StatusCode::SERVICE_UNAVAILABLE).is_transient());
        assert!(!http(StatusCode::UNAUTHORIZED).is_transient());
        assert!(Error::Timeout.is_transient());
        assert!(Error::Network("reset".to_string()).is_transient());
        assert!(!Error::Decode("bad json".to_string()).is_transient());
    }
}
