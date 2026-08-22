use super::DownloadError;

#[test]
fn only_transient_transfer_failures_are_retryable() {
    assert!(DownloadError::Transport("connection reset".to_string()).is_retryable_transfer_failure());
    assert!(DownloadError::HttpStatus(429).is_retryable_transfer_failure());
    assert!(DownloadError::HttpStatus(500).is_retryable_transfer_failure());
    assert!(DownloadError::HttpStatus(599).is_retryable_transfer_failure());

    for error in [
        DownloadError::AuthenticationRequired,
        DownloadError::AccessDenied,
        DownloadError::SourceNotFound,
        DownloadError::SourceGone,
        DownloadError::HttpStatus(400),
        DownloadError::HttpStatus(300),
        DownloadError::Io("disk full".to_string()),
    ] {
        assert!(!error.is_retryable_transfer_failure(), "unexpectedly retryable: {error}");
    }
}
