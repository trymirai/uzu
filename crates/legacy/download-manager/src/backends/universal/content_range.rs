use reqwest::header::HeaderValue;

use crate::backends::universal::UniversalBackendError;

pub struct ContentRange {
    pub start: Option<u64>,
    pub total: Option<u64>,
}

impl ContentRange {
    pub fn parse(header: Option<&HeaderValue>) -> Result<Self, UniversalBackendError> {
        let value = header
            .ok_or_else(|| {
                UniversalBackendError::Protocol("server returned 206 without Content-Range header".to_string())
            })?
            .to_str()
            .map_err(|error| UniversalBackendError::Protocol(format!("non-utf8 Content-Range header: {error}")))?;
        let malformed =
            || UniversalBackendError::Protocol(format!("server returned malformed Content-Range header: {value}"));
        let (range, total) =
            value.strip_prefix("bytes ").ok_or_else(malformed)?.trim_start().split_once('/').ok_or_else(malformed)?;
        Ok(Self {
            start: range.split_once('-').and_then(|(start, _)| start.parse().ok()),
            total: total.parse().ok(),
        })
    }
}
