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
}

impl RequestInfo {
    pub fn new(
        method: Method,
        uri: String,
    ) -> Self {
        Self {
            method,
            uri,
            id: Uuid::new_v4().simple().to_string(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_secs() as i64)
                .unwrap_or(0),
        }
    }

    pub fn id_short(&self) -> String {
        self.id.chars().take(8).collect()
    }
}

#[rocket::async_trait]
impl<'a> FromRequest<'a> for &'a RequestInfo {
    type Error = Infallible;

    async fn from_request(request: &'a Request<'_>) -> Outcome<Self, Self::Error> {
        Outcome::Success(request.local_cache(|| RequestInfo::new(request.method(), request.uri().to_string())))
    }
}
