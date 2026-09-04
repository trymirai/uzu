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
mod tests {
    use reqwest::StatusCode;

    use super::IsTransient;

    #[test]
    fn statuses_are_classified() {
        for status in [
            StatusCode::REQUEST_TIMEOUT,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::GATEWAY_TIMEOUT,
        ] {
            assert!(status.is_transient(), "{status} should be transient");
        }
        for status in [
            StatusCode::OK,
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::NOT_FOUND,
            StatusCode::CONFLICT,
            StatusCode::NOT_IMPLEMENTED,
        ] {
            assert!(!status.is_transient(), "{status} should be fatal");
        }
    }

    #[tokio::test]
    async fn transport_errors_are_classified() {
        let connect = reqwest::Client::new().get("http://127.0.0.1:1").send().await.unwrap_err();
        assert!(connect.is_connect() && connect.is_transient(), "connect should retry: {connect:?}");

        let builder = reqwest::Client::new().get("not-a-url").send().await.unwrap_err();
        assert!(builder.is_builder() && !builder.is_transient(), "builder should be fatal: {builder:?}");
    }
}
