#![allow(dead_code)]

use std::{path::PathBuf, time::Duration};

use futures_util::StreamExt;
use shoji::types::{
    basic::{File, Hash, HashMethod, Metadata},
    model::{Model, ModelAccessibility, ModelBackend, ModelReference, ModelRegistry},
};
use uzu::{
    device::Device,
    storage::{Config, Storage, types::DownloadPhase},
};

use crate::common::mock_download_server::{MockDownloadServer, MockFile, MockFileSet, RouteBehavior};

pub struct StorageFixture {
    pub server: MockDownloadServer,
    pub storage: Storage,
    pub model: Model,
    pub file_set: MockFileSet,
    pub base_path: PathBuf,
    temp_dir_guard: tempfile::TempDir,
}

impl StorageFixture {
    pub async fn new(behavior: RouteBehavior) -> Self {
        let server = MockDownloadServer::start().await;
        let file_set = MockFileSet::qwen_like("uzu-storage/model");
        server.serve_file_set(&file_set, behavior).await;
        Self::from_server(server, file_set).await
    }

    pub async fn with_tokenizer_behavior(behavior: RouteBehavior) -> Self {
        let server = MockDownloadServer::start().await;
        let file_set = MockFileSet::qwen_like("uzu-storage/model");
        server.serve_file(file_set.config.clone(), RouteBehavior::Normal).await;
        server.serve_file(file_set.tokenizer.clone(), behavior).await;
        server.serve_file(file_set.tokenizer_config.clone(), RouteBehavior::Normal).await;
        Self::from_server(server, file_set).await
    }

    pub async fn recreate_storage(&self) -> Storage {
        let file_set = self.file_set.clone();
        let model = model_from_file_set(&self.server, &file_set);
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
        for file in self.file_set.all() {
            let path = self.cache_model_path().join(&file.name);
            let downloaded_bytes = tokio::fs::read(&path)
                .await
                .unwrap_or_else(|error| panic!("failed to read {}: {}", path.display(), error));
            assert_eq!(downloaded_bytes, file.bytes.to_vec(), "downloaded bytes mismatch for {}", file.name);
        }
    }

    async fn from_server(
        server: MockDownloadServer,
        file_set: MockFileSet,
    ) -> Self {
        let temp_dir_guard = tempfile::tempdir().expect("failed to create storage temp dir");
        let base_path = temp_dir_guard.path().to_path_buf();
        let model = model_from_file_set(&server, &file_set);
        let device = Device::new().expect("failed to create device");
        let config = Config::new(device, Some(base_path.clone()), "test_storage".to_string());
        let storage = Storage::new(tokio::runtime::Handle::current(), config).await.expect("failed to create storage");
        storage.refresh(vec![model.clone()]).await.expect("failed to refresh storage");
        Self {
            server,
            storage,
            model,
            file_set,
            base_path,
            temp_dir_guard,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoragePhaseKind {
    NotDownloaded,
    Downloading,
    Paused,
    Downloaded,
    Error,
}

impl StoragePhaseKind {
    fn matches(
        self,
        phase: &DownloadPhase,
    ) -> bool {
        match self {
            Self::NotDownloaded => matches!(phase, DownloadPhase::NotDownloaded {}),
            Self::Downloading => matches!(phase, DownloadPhase::Downloading {}),
            Self::Paused => matches!(phase, DownloadPhase::Paused {}),
            Self::Downloaded => matches!(phase, DownloadPhase::Downloaded {}),
            Self::Error => matches!(phase, DownloadPhase::Error { .. }),
        }
    }
}

fn model_from_file_set(
    server: &MockDownloadServer,
    file_set: &MockFileSet,
) -> Model {
    let registry = ModelRegistry {
        identifier: "test-registry".to_string(),
        metadata: Metadata::external("test-registry".to_string()),
    };
    let backend = ModelBackend {
        identifier: "test-backend".to_string(),
        version: "1.0".to_string(),
        metadata: Metadata::external("test-backend".to_string()),
    };
    Model {
        identifier: "test-model".to_string(),
        registry,
        backends: vec![backend],
        family: None,
        properties: None,
        quantization: None,
        specializations: vec![],
        accessibility: ModelAccessibility::Local {
            reference: ModelReference::Mirai {
                toolchain_version: "1.0".to_string(),
                repository: None,
                source_repository: None,
                files: file_set.all().iter().map(|file| shoji_file_from_mock_file(server, file)).collect(),
            },
        },
    }
}

fn shoji_file_from_mock_file(
    server: &MockDownloadServer,
    file: &MockFile,
) -> File {
    File {
        url: server.url_for_file(file),
        name: file.name.clone(),
        size: file.size as i64,
        hashes: vec![Hash {
            method: HashMethod::CRC32C,
            value: file.crc32c.clone(),
        }],
    }
}
