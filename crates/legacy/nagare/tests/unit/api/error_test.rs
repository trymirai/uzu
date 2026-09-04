use reqwest::StatusCode;

use super::Error;
use crate::api::IsTransient;

fn http(code: StatusCode) -> Error {
    Error::Http {
        code,
        body: String::new(),
    }
}

/// The status list itself is covered on `StatusCode`; this pins the delegation
/// and the variants that carry no status.
#[test]
fn variants_are_classified() {
    assert!(http(StatusCode::SERVICE_UNAVAILABLE).is_transient());
    assert!(!http(StatusCode::UNAUTHORIZED).is_transient());
    assert!(Error::Timeout.is_transient());
    assert!(Error::Network("reset".to_string()).is_transient());
    assert!(!Error::Decode("bad json".to_string()).is_transient());
}
