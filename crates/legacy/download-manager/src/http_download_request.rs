use std::{
    fmt,
    net::IpAddr,
    sync::{Arc, Weak},
};

use http::uri::Scheme;
use reqwest::{RequestBuilder, Url};

use crate::DownloadError;

#[derive(Clone)]
pub struct HttpDownloadRequest {
    pub url: String,
    bearer_token: Option<Weak<str>>,
}

impl HttpDownloadRequest {
    pub fn get(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            bearer_token: None,
        }
    }

    pub fn with_bearer_token(
        url: impl Into<String>,
        bearer_token: &Arc<str>,
    ) -> Self {
        Self {
            url: url.into(),
            bearer_token: Some(Arc::downgrade(bearer_token)),
        }
    }

    pub fn is_authenticated(&self) -> bool {
        self.bearer_token.is_some()
    }

    pub fn bearer_token(&self) -> Result<Option<Arc<str>>, DownloadError> {
        self.bearer_token
            .as_ref()
            .map(|token| token.upgrade().ok_or(DownloadError::AuthenticationUnavailable))
            .transpose()
    }

    pub fn apply(
        &self,
        request: RequestBuilder,
    ) -> Result<RequestBuilder, DownloadError> {
        Ok(match self.bearer_token()? {
            Some(token) => request.bearer_auth(token.as_ref()),
            None => request,
        })
    }

    pub fn validate(&self) -> Result<(), DownloadError> {
        let url = Url::parse(&self.url).map_err(|_| DownloadError::BadUrl)?;
        let scheme = Scheme::try_from(url.scheme()).map_err(|_| DownloadError::BadUrl)?;
        if self.is_authenticated() && scheme != Scheme::HTTPS && !is_loopback(&url) {
            return Err(DownloadError::InsecureAuthenticatedRequest);
        }
        Ok(())
    }
}

impl fmt::Debug for HttpDownloadRequest {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let end = self.url.find(['?', '#']).unwrap_or(self.url.len());
        formatter
            .debug_struct("HttpDownloadRequest")
            .field("url", &&self.url[..end])
            .field("authenticated", &self.is_authenticated())
            .finish()
    }
}

impl PartialEq for HttpDownloadRequest {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.url == other.url
            && match (&self.bearer_token, &other.bearer_token) {
                (Some(left), Some(right)) => Weak::ptr_eq(left, right),
                (None, None) => true,
                _ => false,
            }
    }
}

impl Eq for HttpDownloadRequest {}

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

impl From<&String> for HttpDownloadRequest {
    fn from(url: &String) -> Self {
        Self::get(url)
    }
}

fn is_loopback(url: &Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    host.eq_ignore_ascii_case("localhost") || host.parse::<IpAddr>().is_ok_and(|address| address.is_loopback())
}
