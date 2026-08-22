use super::{HttpDownloadRequest, RequestHeaders};

#[test]
fn debug_redacts_bearer_token() {
    let headers = RequestHeaders::bearer("secret-token").expect("valid header");
    let debug = format!("{headers:?}");

    assert!(!debug.contains("secret-token"), "debug output leaked the token: {debug}");
}

#[test]
fn request_debug_redacts_bearer_token() {
    let request = HttpDownloadRequest::with_headers(
        "https://example.test/model",
        RequestHeaders::bearer("secret-token").expect("valid header"),
    );
    let debug = format!("{request:?}");

    assert!(debug.contains("authorization"));
    assert!(!debug.contains("secret-token"), "debug output leaked the token: {debug}");
}

#[test]
fn rejects_authenticated_plaintext_except_for_loopback_tests() {
    let headers = RequestHeaders::bearer("secret-token").unwrap();
    assert_eq!(
        HttpDownloadRequest::with_headers("http://example.test/model", headers.clone()).validate(),
        Err(crate::DownloadError::InsecureAuthenticatedRequest)
    );
    assert!(HttpDownloadRequest::with_headers("http://127.0.0.1/model", headers).validate().is_ok());
}
