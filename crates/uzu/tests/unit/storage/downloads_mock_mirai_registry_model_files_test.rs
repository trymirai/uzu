use std::time::Duration;

use nagare::api::BaseUrl;
use shoji::traits::Registry;
use tokio::time::{sleep as tokio_sleep, timeout as tokio_timeout};
use uzu::{
    device::Device,
    registry::mirai::{Backend, Config as MiraiRegistryConfig, Registry as MiraiRegistry},
    storage::{Config as StorageConfig, Storage, types::DownloadPhase},
};

use crate::common::{mock_download_server::RouteBehavior, mock_mirai_server::MockMiraiServer};

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_downloads_files_from_mock_mirai_registry() {
    let server = MockMiraiServer::start(RouteBehavior::Normal).await;
    let device = Device::new().expect("failed to create device");
    let registry = MiraiRegistry::new(MiraiRegistryConfig {
        api_key: None,
        base_url: BaseUrl::Custom(server.api_base_url()),
        device: device.clone(),
        backends: vec![Backend {
            identifier: "mirai".to_string(),
            version: "0.1.9".to_string(),
        }],
        include_traces: false,
    })
    .expect("failed to create registry");
    let models = registry.models().await.expect("mock registry should return models");
    let model = models.first().cloned().expect("mock registry should include a model");

    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let storage_config = StorageConfig::new(device, Some(temp_dir.path().to_path_buf()), "mock_mirai_storage".to_string());
    let storage = Storage::new(tokio::runtime::Handle::current(), storage_config)
        .await
        .expect("failed to create storage");

    storage.refresh(models).await.expect("failed to refresh storage from mock registry models");
    storage.download(&model.identifier).await.expect("failed to start model download");

    let downloaded = tokio_timeout(Duration::from_secs(20), async {
        loop {
            let state = storage.state(&model.identifier).await.expect("model should have storage state");
            if matches!(state.phase, DownloadPhase::Downloaded {}) {
                return state;
            }
            tokio_sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("timed out waiting for model download");

    assert_eq!(downloaded.downloaded_bytes, downloaded.total_bytes);
    let cache_path = storage.config.cache_model_path(&model).expect("model should have cache path");
    for payload in server.payloads() {
        let file_path = cache_path.join(&payload.file.name);
        let downloaded_bytes =
            tokio::fs::read(&file_path).await.unwrap_or_else(|error| panic!("{}: {}", file_path.display(), error));
        assert_eq!(downloaded_bytes, payload.bytes.to_vec());
    }
}
