use reqwest::StatusCode;

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
    fn is_transient(&self) -> bool {
        self.is_timeout() || self.is_connect()
    }
}
