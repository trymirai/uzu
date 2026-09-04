use reqwest::StatusCode;

/// Whether a failure is worth retrying.
pub trait IsTransient {
    fn is_transient(&self) -> bool;
}

impl IsTransient for StatusCode {
    fn is_transient(&self) -> bool {
        matches!(
            *self,
            StatusCode::REQUEST_TIMEOUT
                | StatusCode::TOO_MANY_REQUESTS
                | StatusCode::INTERNAL_SERVER_ERROR
                | StatusCode::BAD_GATEWAY
                | StatusCode::SERVICE_UNAVAILABLE
                | StatusCode::GATEWAY_TIMEOUT
        )
    }
}

impl IsTransient for reqwest::Error {
    /// Only timeouts and connect failures. A body, decode, redirect or builder
    /// failure means the request reached a verdict, so replaying it is pointless.
    fn is_transient(&self) -> bool {
        self.is_timeout() || self.is_connect()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/api/is_transient_test.rs"]
mod tests;
