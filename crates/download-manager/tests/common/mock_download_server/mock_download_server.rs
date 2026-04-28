use wiremock::{
    Mock, MockServer, Request, Respond, ResponseTemplate,
    matchers::{method, path as wiremock_path},
};

use crate::common::mock_download_server::{FilePayload, RegistryFixture, RouteBehavior};

pub struct MockDownloadServer {
    server: MockServer,
}

impl MockDownloadServer {
    pub async fn start() -> Self {
        Self {
            server: MockServer::start().await,
        }
    }

    pub fn base_url(&self) -> String {
        self.server.uri()
    }

    pub fn url_for_path(
        &self,
        path: &str,
    ) -> String {
        format!("{}{}", self.base_url(), path)
    }

    pub async fn serve_file(
        &self,
        payload: FilePayload,
        behavior: RouteBehavior,
    ) {
        let file_path = payload.path();
        Mock::given(method("GET"))
            .and(wiremock_path(file_path.clone()))
            .respond_with(FileResponder {
                payload: payload.clone(),
                behavior: behavior.clone(),
                include_body: true,
            })
            .mount(&self.server)
            .await;
        Mock::given(method("HEAD"))
            .and(wiremock_path(file_path))
            .respond_with(FileResponder {
                payload,
                behavior,
                include_body: false,
            })
            .mount(&self.server)
            .await;
    }

    pub async fn serve_registry_fixture(
        &self,
        fixture: &RegistryFixture,
    ) {
        for payload in fixture.payloads() {
            self.serve_file(payload, RouteBehavior::Normal).await;
        }
    }

    pub async fn serve_json(
        &self,
        path: &str,
        json: String,
    ) {
        Mock::given(method("POST"))
            .and(wiremock_path(path))
            .respond_with(ResponseTemplate::new(200).set_body_raw(json, "application/json"))
            .mount(&self.server)
            .await;
    }
}

#[derive(Clone)]
struct FileResponder {
    payload: FilePayload,
    behavior: RouteBehavior,
    include_body: bool,
}

impl Respond for FileResponder {
    fn respond(
        &self,
        request: &Request,
    ) -> ResponseTemplate {
        let bytes = match self.behavior {
            RouteBehavior::CorruptBody => self.payload.corrupt_bytes(),
            RouteBehavior::Normal => self.payload.bytes.clone(),
        };
        let file_size = bytes.len() as u64;
        let (status, body_start, body_end_exclusive) =
            parse_range(request.headers.get("range").and_then(|header| header.to_str().ok()), file_size)
                .unwrap_or((200, 0, file_size));
        let body = if self.include_body {
            bytes[body_start as usize..body_end_exclusive as usize].to_vec()
        } else {
            Vec::new()
        };
        let content_length = if self.include_body {
            body.len() as u64
        } else {
            body_end_exclusive.saturating_sub(body_start)
        };

        let mut response = ResponseTemplate::new(status)
            .set_body_bytes(body)
            .insert_header("Accept-Ranges", "bytes")
            .insert_header("Last-Modified", self.payload.last_modified.clone())
            .insert_header("Content-Length", content_length.to_string());
        if status == 206 {
            response = response.insert_header(
                "Content-Range",
                format!("bytes {}-{}/{}", body_start, body_end_exclusive.saturating_sub(1), file_size),
            );
        }
        response
    }
}

fn parse_range(
    header: Option<&str>,
    file_size: u64,
) -> Option<(u16, u64, u64)> {
    let range = header?.strip_prefix("bytes=")?;
    let (start, end) = range.split_once('-')?;
    let start = start.parse::<u64>().ok()?;
    if start >= file_size {
        return Some((416, 0, 0));
    }
    let end_exclusive = end.parse::<u64>().ok().map_or(file_size, |end| (end + 1).min(file_size));
    Some((206, start, end_exclusive))
}
