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
