use serde_json::json;
use shoji::types::model::Model;

use crate::common::mock_download_server::{FilePayload, MockDownloadServer, RegistryFixture};

pub struct MockMiraiServer {
    server: MockDownloadServer,
    registry_fixture: RegistryFixture,
}

impl MockMiraiServer {
    pub async fn start() -> Self {
        let server = MockDownloadServer::start().await;
        let registry_fixture = RegistryFixture::llama_3_2_1b_instruct(&server.base_url(), "mock-mirai");
        server.serve_registry_fixture(&registry_fixture).await;
        let model = registry_fixture.model.clone();
        let registry_response = registry_response_json(&model);
        server.serve_json("/api/v1/fetch/models", registry_response).await;

        Self {
            server,
            registry_fixture,
        }
    }

    pub fn api_base_url(&self) -> String {
        format!("{}/api/v1", self.server.base_url())
    }

    pub fn payloads(&self) -> Box<[FilePayload]> {
        self.registry_fixture.payloads()
    }
}

fn registry_response_json(model: &Model) -> String {
    let registry_metadata = model.registry.metadata.clone();
    let backend_metadatas = model.backends.iter().map(|backend| backend.metadata.clone()).collect::<Vec<_>>();
    let family_metadata = model.family.as_ref().map(|family| family.metadata.clone());
    let vendor_metadata = model.family.as_ref().map(|family| family.vendor.metadata.clone());

    let mut metadatas = vec![registry_metadata.clone()];
    metadatas.extend(backend_metadatas.iter().cloned());
    if let Some(metadata) = family_metadata.clone() {
        metadatas.push(metadata);
    }
    if let Some(metadata) = vendor_metadata.clone() {
        metadatas.push(metadata);
    }

    let family = model.family.as_ref().map(|family| {
        json!({
            "id": family.identifier,
            "vendor": {
                "id": family.vendor.identifier,
                "metadata_id": family.vendor.metadata.identifier,
            },
            "metadata_id": family.metadata.identifier,
        })
    });

    let response = json!({
        "models": [{
            "id": model.identifier,
            "registry": {
                "id": model.registry.identifier,
                "metadata_id": model.registry.metadata.identifier,
            },
            "backends": model.backends.iter().map(|backend| {
                json!({
                    "id": backend.identifier,
                    "version": backend.version,
                    "metadata_id": backend.metadata.identifier,
                })
            }).collect::<Vec<_>>(),
            "family": family,
            "properties": model.properties,
            "quantization": model.quantization,
            "specializations": model.specializations,
            "accessibility": model.accessibility,
        }],
        "metadatas": metadatas,
    });

    serde_json::to_string(&response).unwrap()
}
