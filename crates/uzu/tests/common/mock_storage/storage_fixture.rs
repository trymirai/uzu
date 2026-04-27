use std::{path::PathBuf, time::Duration};

use futures_util::StreamExt;
use shoji::types::model::Model;
use uzu::{
    device::Device,
    storage::{Config, Storage},
};

use crate::common::{
    mock_download_server::{MockDownloadServer, RegistryFixture, RouteBehavior},
    mock_storage::StoragePhaseKind,
};

pub struct StorageFixture {
    pub server: MockDownloadServer,
    pub storage: Storage,
    pub model: Model,
    pub registry_fixture: RegistryFixture,
    pub base_path: PathBuf,
    temp_dir_guard: tempfile::TempDir,
}

impl StorageFixture {
    pub async fn new(behavior: RouteBehavior) -> Self {
        let server = MockDownloadServer::start().await;
        let registry_fixture = RegistryFixture::llama_3_2_1b_instruct(&server.base_url(), "uzu-storage");
        server.serve_registry_fixture(&registry_fixture, behavior).await;
        Self::from_server(server, registry_fixture).await
    }

    pub async fn with_tokenizer_behavior(behavior: RouteBehavior) -> Self {
        let server = MockDownloadServer::start().await;
        let registry_fixture = RegistryFixture::llama_3_2_1b_instruct(&server.base_url(), "uzu-storage");
        for payload in registry_fixture.payloads() {
            let route_behavior = if payload.file.name == "tokenizer.json" {
                behavior.clone()
            } else {
                RouteBehavior::Normal
            };
            server.serve_file(payload, route_behavior).await;
        }
        Self::from_server(server, registry_fixture).await
    }

    pub async fn recreate_storage(&self) -> Storage {
        let model = self.registry_fixture.model.clone();
        let device = Device::new().expect("failed to create device");
        let config = Config::new(device, Some(self.base_path.clone()), "test_storage".to_string());
        let storage =
            Storage::new(tokio::runtime::Handle::current(), config).await.expect("failed to recreate storage");
        storage.refresh(vec![model.clone()]).await.expect("failed to refresh storage");
        storage
    }

    pub async fn item(&self) -> uzu::storage::types::Item {
        self.storage.get(&self.model.identifier).await.expect("model should be in storage")
    }

    pub async fn download(&self) {
        self.storage.download(&self.model.identifier).await.expect("failed to start model download");
    }

    pub async fn pause(&self) {
        self.storage.pause(&self.model.identifier).await.expect("failed to pause model download");
    }

    pub async fn delete(&self) {
        self.storage.delete(&self.model.identifier).await.expect("failed to delete model download");
    }

    pub async fn wait_for_phase(
        &self,
        phase_kind: StoragePhaseKind,
    ) -> uzu::storage::types::DownloadState {
        tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                let state = self.storage.state(&self.model.identifier).await.expect("model state should exist");
                if phase_kind.matches(&state.phase) {
                    return state;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {:?} on {}", phase_kind, self.model.identifier))
    }

    pub async fn wait_for_broadcast_phase(
        &self,
        phase_kind: StoragePhaseKind,
    ) -> uzu::storage::types::DownloadState {
        let mut updates = self.storage.subscribe();
        tokio::time::timeout(Duration::from_secs(20), async {
            while let Some(Ok((identifier, state))) = updates.next().await {
                if identifier == self.model.identifier && phase_kind.matches(&state.phase) {
                    return state;
                }
            }
            panic!("storage broadcast stream ended before {:?}", phase_kind);
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for broadcast {:?}", phase_kind))
    }

    pub fn cache_model_path(&self) -> PathBuf {
        self.storage.config.cache_model_path(&self.model).expect("model should have cache path")
    }

    pub async fn assert_files_match_server(&self) {
        for payload in self.registry_fixture.payloads() {
            let path = self.cache_model_path().join(&payload.file.name);
            let downloaded_bytes = tokio::fs::read(&path)
                .await
                .unwrap_or_else(|error| panic!("failed to read {}: {}", path.display(), error));
            assert_eq!(downloaded_bytes, payload.bytes.to_vec(), "downloaded bytes mismatch for {}", payload.file.name);
        }
    }

    async fn from_server(
        server: MockDownloadServer,
        registry_fixture: RegistryFixture,
    ) -> Self {
        let temp_dir_guard = tempfile::tempdir().expect("failed to create storage temp dir");
        let base_path = temp_dir_guard.path().to_path_buf();
        let model = registry_fixture.model.clone();
        let device = Device::new().expect("failed to create device");
        let config = Config::new(device, Some(base_path.clone()), "test_storage".to_string());
        let storage = Storage::new(tokio::runtime::Handle::current(), config).await.expect("failed to create storage");
        storage.refresh(vec![model.clone()]).await.expect("failed to refresh storage");
        Self {
            server,
            storage,
            model,
            registry_fixture,
            base_path,
            temp_dir_guard,
        }
    }
}
