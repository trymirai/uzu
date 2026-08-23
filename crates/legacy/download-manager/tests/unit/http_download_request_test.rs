use super::{HttpDownloadRequest, RequestHeaders, is_transport_downgrade};

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

#[test]
fn any_redirect_leaving_https_is_a_downgrade() {
    assert!(is_transport_downgrade(Some("https"), Some("http")));
    assert!(is_transport_downgrade(Some("https"), Some("ftp")));
    assert!(is_transport_downgrade(Some("https"), None));
    assert!(!is_transport_downgrade(Some("https"), Some("https")));
    assert!(!is_transport_downgrade(Some("http"), Some("http")));
    assert!(!is_transport_downgrade(None, Some("http")));
}
