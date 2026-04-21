use crate::prelude::*;

/// Common NSURLError codes from the NSURLErrorDomain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum URLSessionErrorCode {
    /// -999: Request was cancelled (expected during pause operations)
    Cancelled,
    /// -1000: Malformed URL
    BadURL,
    /// -1001: Connection timeout
    TimedOut,
    /// -1002: Unsupported URL scheme
    UnsupportedURL,
    /// -1003: DNS resolution failed
    CannotFindHost,
    /// -1004: Connection refused by server
    CannotConnectToHost,
    /// -1005: Connection dropped during transfer
    NetworkConnectionLost,
    /// -1006: DNS lookup failed
    DNSLookupFailed,
    /// -1007: Too many redirects
    HTTPTooManyRedirects,
    /// -1008: Resource unavailable
    ResourceUnavailable,
    /// -1009: No internet connection
    NotConnectedToInternet,
    /// -1010: Redirect to non-existent location
    RedirectToNonExistentLocation,
    /// -1011: Bad server response
    BadServerResponse,
    /// -1012: User cancelled authentication
    UserCancelledAuthentication,
    /// -1013: Authentication required
    UserAuthenticationRequired,
    /// -1200: Secure connection failed
    SecureConnectionFailed,
    /// -1201: Server certificate invalid
    ServerCertificateHasBadDate,
    /// -1202: Server certificate untrusted
    ServerCertificateUntrusted,
    /// -1203: Server certificate has unknown root
    ServerCertificateHasUnknownRoot,
    /// -1204: Server certificate not yet valid
    ServerCertificateNotYetValid,
    /// -1205: Client certificate rejected
    ClientCertificateRejected,
    /// -1206: Client certificate required
    ClientCertificateRequired,
    /// -2000: Cannot load from network
    CannotLoadFromNetwork,
    /// Other error code not in the known list
    Other(i64),
}

impl URLSessionErrorCode {
    /// Parse an error code from NSError
    pub fn from_error_code(code: i64) -> Self {
        match code {
            -999 => Self::Cancelled,
            -1000 => Self::BadURL,
            -1001 => Self::TimedOut,
            -1002 => Self::UnsupportedURL,
            -1003 => Self::CannotFindHost,
            -1004 => Self::CannotConnectToHost,
            -1005 => Self::NetworkConnectionLost,
            -1006 => Self::DNSLookupFailed,
            -1007 => Self::HTTPTooManyRedirects,
            -1008 => Self::ResourceUnavailable,
            -1009 => Self::NotConnectedToInternet,
            -1010 => Self::RedirectToNonExistentLocation,
            -1011 => Self::BadServerResponse,
            -1012 => Self::UserCancelledAuthentication,
            -1013 => Self::UserAuthenticationRequired,
            -1200 => Self::SecureConnectionFailed,
            -1201 => Self::ServerCertificateHasBadDate,
            -1202 => Self::ServerCertificateUntrusted,
            -1203 => Self::ServerCertificateHasUnknownRoot,
            -1204 => Self::ServerCertificateNotYetValid,
            -1205 => Self::ClientCertificateRejected,
            -1206 => Self::ClientCertificateRequired,
            -2000 => Self::CannotLoadFromNetwork,
            other => Self::Other(other),
        }
    }

    /// Get the raw error code value
    pub fn code(&self) -> i64 {
        match self {
            Self::Cancelled => -999,
            Self::BadURL => -1000,
            Self::TimedOut => -1001,
            Self::UnsupportedURL => -1002,
            Self::CannotFindHost => -1003,
            Self::CannotConnectToHost => -1004,
            Self::NetworkConnectionLost => -1005,
            Self::DNSLookupFailed => -1006,
            Self::HTTPTooManyRedirects => -1007,
            Self::ResourceUnavailable => -1008,
            Self::NotConnectedToInternet => -1009,
            Self::RedirectToNonExistentLocation => -1010,
            Self::BadServerResponse => -1011,
            Self::UserCancelledAuthentication => -1012,
            Self::UserAuthenticationRequired => -1013,
            Self::SecureConnectionFailed => -1200,
            Self::ServerCertificateHasBadDate => -1201,
            Self::ServerCertificateUntrusted => -1202,
            Self::ServerCertificateHasUnknownRoot => -1203,
            Self::ServerCertificateNotYetValid => -1204,
            Self::ClientCertificateRejected => -1205,
            Self::ClientCertificateRequired => -1206,
            Self::CannotLoadFromNetwork => -2000,
            Self::Other(code) => *code,
        }
    }

    /// Get a human-readable description of the error
    pub fn description(&self) -> &'static str {
        match self {
            Self::Cancelled => "Request was cancelled",
            Self::BadURL => "Malformed URL",
            Self::TimedOut => "Connection timeout",
            Self::UnsupportedURL => "Unsupported URL scheme",
            Self::CannotFindHost => "DNS resolution failed",
            Self::CannotConnectToHost => "Connection refused by server",
            Self::NetworkConnectionLost => "Connection dropped during transfer",
            Self::DNSLookupFailed => "DNS lookup failed",
            Self::HTTPTooManyRedirects => "Too many redirects",
            Self::ResourceUnavailable => "Resource unavailable",
            Self::NotConnectedToInternet => "No internet connection",
            Self::RedirectToNonExistentLocation => "Redirect to non-existent location",
            Self::BadServerResponse => "Bad server response",
            Self::UserCancelledAuthentication => "User cancelled authentication",
            Self::UserAuthenticationRequired => "Authentication required",
            Self::SecureConnectionFailed => "Secure connection failed",
            Self::ServerCertificateHasBadDate => "Server certificate has invalid date",
            Self::ServerCertificateUntrusted => "Server certificate is untrusted",
            Self::ServerCertificateHasUnknownRoot => "Server certificate has unknown root",
            Self::ServerCertificateNotYetValid => "Server certificate not yet valid",
            Self::ClientCertificateRejected => "Client certificate rejected",
            Self::ClientCertificateRequired => "Client certificate required",
            Self::CannotLoadFromNetwork => "Cannot load from network",
            Self::Other(_) => "Unknown error",
        }
    }

    /// Check if this error should be ignored (e.g., expected cancellations)
    pub fn should_ignore(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

/// Parsed error information from NSError
#[derive(Debug, Clone)]
pub struct URLSessionError {
    /// Parsed error code
    pub code: URLSessionErrorCode,
    /// Localized error description from the system
    pub localized_description: String,
}

impl URLSessionError {
    /// Parse an NSError into a URLSessionError
    pub fn from_nserror(error: &NSError) -> Self {
        let code = URLSessionErrorCode::from_error_code(error.code() as i64);
        let localized_description = error.localizedDescription().to_string();

        Self {
            code,
            localized_description,
        }
    }

    /// Get a formatted error message suitable for user display
    pub fn user_message(&self) -> String {
        // Use localized description if available and non-empty
        if !self.localized_description.is_empty() {
            format!("{} (code: {})", self.localized_description, self.code.code())
        } else {
            format!("{} (code: {})", self.code.description(), self.code.code())
        }
    }

    /// Check if this error should be ignored
    pub fn should_ignore(&self) -> bool {
        self.code.should_ignore()
    }
}
