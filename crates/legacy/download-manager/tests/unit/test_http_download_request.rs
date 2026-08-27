use std::sync::Arc;

use crate::{DownloadError, HttpDownloadRequest};

#[test]
fn request_debug_redacts_bearer_token() {
    let token: Arc<str> = Arc::from("secret-token");
    let request = HttpDownloadRequest::with_bearer_token("https://example.com/model?signature=signed-secret", &token);
    let debug = format!("{request:?}");

    assert!(debug.contains("authenticated: true"));
    assert!(!debug.contains("secret-token"));
    assert!(!debug.contains("signed-secret"));
}

#[test]
fn authenticated_request_requires_https_except_for_loopback() {
    let token: Arc<str> = Arc::from("token");
    let insecure = HttpDownloadRequest::with_bearer_token("http://example.com/model", &token);
    let loopback = HttpDownloadRequest::with_bearer_token("http://127.0.0.1/model", &token);

    assert_eq!(insecure.validate(), Err(DownloadError::InsecureAuthenticatedRequest));
    assert_eq!(loopback.validate(), Ok(()));
}

#[test]
fn request_does_not_keep_credentials_alive() {
    let token: Arc<str> = Arc::from("token");
    let request = HttpDownloadRequest::with_bearer_token("https://example.com/model", &token);
    drop(token);

    assert_eq!(request.bearer_token(), Err(DownloadError::AuthenticationUnavailable));
}

#[test]
fn authenticated_requests_match_only_the_same_credential_owner() {
    let first_token: Arc<str> = Arc::from("token");
    let second_token: Arc<str> = Arc::from("token");
    let first = HttpDownloadRequest::with_bearer_token("https://example.com/model", &first_token);
    let first_clone = HttpDownloadRequest::with_bearer_token("https://example.com/model", &first_token);
    let second = HttpDownloadRequest::with_bearer_token("https://example.com/model", &second_token);

    assert_eq!(first, first_clone);
    assert_ne!(first, second);
}
