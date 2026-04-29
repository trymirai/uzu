use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};
use bytesize::ByteSize;
use serde_json::json;
use shoji::types::{
    basic::{File, Hash, HashMethod},
    model::Model,
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

use crate::{
    Behavior, Error, Result, ServedFile, bytes,
    file_server::{FileServer, FileServerTask},
    model,
};

const FILES: &[(&str, ByteSize)] = &[
    ("config.json", ByteSize::b(55_559)),
    ("tokenizer.json", ByteSize::b(17_209_920)),
    ("model.safetensors", ByteSize::mib(32)),
    ("traces.safetensors", ByteSize::b(31_050_840)),
];
const DEFAULT_MODEL_IDENTIFIER: &str = "mock/MockModel-1B";

pub struct MockRegistry {
    server: MockServer,
    _file_server_task: FileServerTask,
    pub models: Box<[Model]>,
    pub files: Box<[ServedFile]>,
}

impl MockRegistry {
    pub async fn start() -> Result<Self> {
        Self::start_with(Behavior::empty()).await
    }

    pub async fn start_with(behavior: Behavior) -> Result<Self> {
        let server = MockServer::start().await;
        let file_server = FileServer::bind().await?;
        let files = build_files(&file_server.base_url)?;
        let models = [model::mock_model(DEFAULT_MODEL_IDENTIFIER, &files)].into();
        let file_server_task = file_server.serve(files.clone(), behavior);
        let registry = Self {
            server,
            _file_server_task: file_server_task,
            models,
            files,
        };
        registry.mount_listing_route().await?;
        Ok(registry)
    }

    pub fn file(
        &self,
        name: &str,
    ) -> Result<&ServedFile> {
        self.files.iter().find(|served_file| served_file.file.name == name).ok_or_else(|| Error::FileNotFound {
            name: name.to_string(),
        })
    }

    async fn mount_listing_route(&self) -> Result<()> {
        Mock::given(method("POST"))
            .and(path("/fetch/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(listing_response(&self.models)?))
            .mount(&self.server)
            .await;
        Ok(())
    }
}

fn build_files(base_url: &str) -> Result<Box<[ServedFile]>> {
    FILES
        .iter()
        .map(|(name, size)| {
            let file_size = size.as_u64();
            let byte_count = usize::try_from(file_size).map_err(|_| Error::FileSizeOutOfRange {
                name: (*name).to_string(),
                size: *size,
                target_type: "usize",
            })?;
            let shoji_file_size = i64::try_from(file_size).map_err(|_| Error::FileSizeOutOfRange {
                name: (*name).to_string(),
                size: *size,
                target_type: "i64",
            })?;
            let bytes = bytes::generate(name, byte_count);
            let crc32c = BASE64_STANDARD.encode(crc32c::crc32c(&bytes).to_be_bytes());
            Ok(ServedFile {
                file: File {
                    url: format!("{}/{}", base_url.trim_end_matches('/'), name),
                    name: (*name).to_string(),
                    size: shoji_file_size,
                    hashes: vec![Hash {
                        method: HashMethod::CRC32C,
                        value: crc32c,
                    }],
                },
                bytes,
            })
        })
        .collect()
}

fn listing_response(models: &[Model]) -> Result<serde_json::Value> {
    let models_json = models
        .iter()
        .map(|model| {
            let registry = &model.registry;
            let backend = model.backends.first().ok_or(Error::MissingBackend)?;
            Ok(json!({
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
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let metadatas = models
        .iter()
        .map(|model| {
            let backend = model.backends.first().ok_or(Error::MissingBackend)?;
            Ok([json!(model.registry.metadata), json!(backend.metadata)])
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    Ok(json!({
        "models": models_json,
        "metadatas": metadatas,
    }))
}
