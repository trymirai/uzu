use std::{
    convert::Infallible,
    time::{SystemTime, UNIX_EPOCH},
};

use rocket::{
    Request,
    http::Method,
    request::{FromRequest, Outcome},
};
use uuid::Uuid;

pub struct RequestInfo {
    pub method: Method,
    pub uri: String,
    pub id: String,
    pub created_at: i64,
    pub span: tracing::Span,
    pub session_id: Option<String>,
}

impl RequestInfo {
    pub fn new(
        method: Method,
        uri: String,
    ) -> Self {
        Self::with_session(method, uri, None)
    }

    pub fn with_session(
        method: Method,
        uri: String,
        session_id: Option<String>,
    ) -> Self {
        let id = Uuid::new_v4().simple().to_string();
        let span = tracing::info_span!(parent: None, "request", request_id = &id[..8]);
        Self {
            method,
            uri,
            id,
            span,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_secs() as i64)
                .unwrap_or(0),
            session_id,
        }
    }
}

#[rocket::async_trait]
impl<'a> FromRequest<'a> for &'a RequestInfo {
    type Error = Infallible;

    async fn from_request(request: &'a Request<'_>) -> Outcome<Self, Self::Error> {
        let session_id = request
            .headers()
            .get_one("x-session-id")
            .or_else(|| request.headers().get_one("X-Session-ID"))
            .map(|s| s.to_string());
        Outcome::Success(request.local_cache(|| RequestInfo::with_session(request.method(), request.uri().to_string(), session_id)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::log::RequestLoggingFairing;
    use rocket::{get, http::Header, local::asynchronous::Client, routes};

    #[get("/echo-session")]
    async fn echo_session(request_info: &RequestInfo) -> String {
        request_info.session_id.clone().unwrap_or_default()
    }

    /// The logging fairing must not pre-seed `RequestInfo` into the
    /// request-local cache; a cached value would shadow this guard and the
    /// `X-Session-ID` header would never be read.
    #[tokio::test]
    async fn guard_reads_session_header_despite_logging_fairing() {
        let rocket = rocket::custom(rocket::Config::debug_default())
            .attach(RequestLoggingFairing)
            .mount("/", routes![echo_session]);
        let client = Client::debug(rocket).await.unwrap();
        let mut request = client.get("/echo-session");
        request.add_header(Header::new("X-Session-ID", "sess_from_header"));
        let body = request.dispatch().await.into_string().await;
        assert_eq!(body.as_deref(), Some("sess_from_header"));
    }
}
