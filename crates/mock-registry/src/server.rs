use std::sync::Arc;

use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};
use bytesize::ByteSize;
use reqwest::Url;
use serde_json::json;
use shoji::types::{
    basic::{File, Hash, HashMethod},
    model::Model,
};
use wiremock::{
    Mock, MockServer, Request, Respond, ResponseTemplate,
    matchers::{method, path},
};

use crate::{Behavior, ServedFile, bytes, model};

const LAST_MODIFIED: &str = "Sun, 26 Apr 2026 12:00:00 GMT";

const FILES: &[(&str, ByteSize)] = &[
    ("config.json", ByteSize::b(55_559)),
    ("tokenizer.json", ByteSize::b(17_209_920)),
    ("model.safetensors", ByteSize::mib(32)),
    ("traces.safetensors", ByteSize::b(31_050_840)),
];

pub struct MockRegistry {
    server: MockServer,
    model: Model,
    files: Box<[ServedFile]>,
}

impl MockRegistry {
    pub async fn start() -> Self {
        Self::start_with(Behavior::Normal).await
    }

    pub async fn start_with(behavior: Behavior) -> Self {
        let server = MockServer::start().await;
        let files = build_files(&server.uri());
        let model = model::mock_model(&files);
        let registry = Self {
            server,
            model,
            files,
        };
        registry.mount_file_routes(behavior).await;
        registry.mount_listing_route().await;
        registry
    }

    pub fn base_url(&self) -> String {
        self.server.uri()
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn files(&self) -> &[ServedFile] {
        &self.files
    }

    pub fn file(
        &self,
        name: &str,
    ) -> &ServedFile {
        self.files.iter().find(|served_file| served_file.file.name == name).expect("mock registry file must exist")
    }

    async fn mount_file_routes(
        &self,
        behavior: Behavior,
    ) {
        for served_file in self.files.iter().cloned() {
            let file_path = path_from_url(&served_file.file.url);
            Mock::given(method("GET"))
                .and(path(file_path.clone()))
                .respond_with(FileResponder {
                    served_file: served_file.clone(),
                    behavior,
                    include_body: true,
                })
                .mount(&self.server)
                .await;
            Mock::given(method("HEAD"))
                .and(path(file_path))
                .respond_with(FileResponder {
                    served_file,
                    behavior,
                    include_body: false,
                })
                .mount(&self.server)
                .await;
        }
    }

    async fn mount_listing_route(&self) {
        Mock::given(method("POST"))
            .and(path("/fetch/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(listing_response(&self.model)))
            .mount(&self.server)
            .await;
    }
}

#[derive(Clone)]
struct FileResponder {
    served_file: ServedFile,
    behavior: Behavior,
    include_body: bool,
}

impl Respond for FileResponder {
    fn respond(
        &self,
        request: &Request,
    ) -> ResponseTemplate {
        let bytes = match self.behavior {
            Behavior::Normal => self.served_file.bytes.clone(),
            Behavior::CorruptBody => corrupt_bytes(&self.served_file.bytes),
        };
        let file_size = bytes.len() as u64;
        let (status, body_start, body_end_exclusive) =
            parse_range(request.headers.get("range").and_then(|header| header.to_str().ok()), file_size)
                .unwrap_or((200, 0, file_size));
        let body = if self.include_body && status != 416 {
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
            .insert_header("Last-Modified", LAST_MODIFIED)
            .insert_header("Content-Length", content_length.to_string());
        if status == 206 {
            response = response.insert_header(
                "Content-Range",
                format!("bytes {}-{}/{}", body_start, body_end_exclusive.saturating_sub(1), file_size),
            );
        } else if status == 416 {
            response = response.insert_header("Content-Range", format!("bytes */{}", file_size));
        }
        response
    }
}

fn build_files(base_url: &str) -> Box<[ServedFile]> {
    FILES
        .iter()
        .map(|(name, size)| {
            let bytes = bytes::generate(name, usize::try_from(size.as_u64()).expect("mock file size must fit usize"));
            let crc32c = BASE64_STANDARD.encode(crc32c::crc32c(&bytes).to_be_bytes());
            ServedFile {
                file: File {
                    url: format!("{}/{}", base_url.trim_end_matches('/'), name),
                    name: (*name).to_string(),
                    size: i64::try_from(size.as_u64()).expect("mock file size must fit i64"),
                    hashes: vec![Hash {
                        method: HashMethod::CRC32C,
                        value: crc32c,
                    }],
                },
                bytes,
            }
        })
        .collect()
}

fn corrupt_bytes(bytes: &Arc<[u8]>) -> Arc<[u8]> {
    let mut corrupted_bytes = bytes.to_vec();
    if let Some(first_byte) = corrupted_bytes.first_mut() {
        *first_byte = first_byte.wrapping_add(1);
    }
    corrupted_bytes.into()
}

fn path_from_url(url: &str) -> String {
    Url::parse(url).expect("mock registry file URL must be valid").path().to_string()
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

fn listing_response(model: &Model) -> serde_json::Value {
    let registry = &model.registry;
    let backend = model.backends.first().expect("mock model must have a backend");
    json!({
        "models": [{
            "id": model.identifier,
            "registry": {
                "id": registry.identifier,
                "metadata_id": registry.metadata.identifier,
            },
            "backends": [{
                "id": backend.identifier,
                "version": backend.version,
                "metadata_id": backend.metadata.identifier,
            }],
            "family": null,
            "properties": null,
            "quantization": null,
            "specializations": [],
            "accessibility": model.accessibility,
        }],
        "metadatas": [
            registry.metadata,
            backend.metadata,
        ],
    })
}
