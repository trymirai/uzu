use std::{fmt, net::IpAddr};

use http::{HeaderMap, HeaderName, HeaderValue, header::AUTHORIZATION};

use crate::DownloadError;

/// HTTP headers attached to a download request.
///
/// Header values are deliberately omitted from [`Debug`] output so bearer
/// tokens cannot leak through configuration logging.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct RequestHeaders(HeaderMap);

impl RequestHeaders {
    pub fn bearer(token: &str) -> Result<Self, DownloadError> {
        let mut value =
            HeaderValue::from_str(&format!("Bearer {token}")).map_err(|_| DownloadError::InvalidRequestHeader)?;
        value.set_sensitive(true);
        Ok(Self::authorization(value))
    }

    pub fn authorization(mut value: HeaderValue) -> Self {
        value.set_sensitive(true);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, value);
        Self(headers)
    }

    pub(crate) fn as_header_map(&self) -> &HeaderMap {
        &self.0
    }

    pub(crate) fn has_authorization(&self) -> bool {
        self.0.contains_key(AUTHORIZATION)
    }
}

impl fmt::Debug for RequestHeaders {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let names = self.0.keys().map(HeaderName::as_str).collect::<Vec<_>>();
        formatter.debug_struct("RequestHeaders").field("names", &names).finish()
    }
}

/// A resolved HTTP source for one file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpDownloadRequest {
    pub url: String,
    pub headers: RequestHeaders,
}

impl HttpDownloadRequest {
    pub fn get(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            headers: RequestHeaders::default(),
        }
    }

    pub fn with_headers(
        url: impl Into<String>,
        headers: RequestHeaders,
    ) -> Self {
        Self {
            url: url.into(),
            headers,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), DownloadError> {
        let url = reqwest::Url::parse(&self.url).map_err(|_| DownloadError::BadUrl)?;
        if self.headers.has_authorization() && url.scheme() != "https" && !is_loopback(&url) {
            return Err(DownloadError::InsecureAuthenticatedRequest);
        }
        Ok(())
    }
}

fn is_loopback(url: &reqwest::Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    host.eq_ignore_ascii_case("localhost") || host.parse::<IpAddr>().is_ok_and(|address| address.is_loopback())
}

impl From<String> for HttpDownloadRequest {
    fn from(url: String) -> Self {
        Self::get(url)
    }
}

impl From<&str> for HttpDownloadRequest {
    fn from(url: &str) -> Self {
        Self::get(url)
    }
}

#[cfg(test)]
#[path = "../tests/unit/http_download_request_test.rs"]
mod tests;
