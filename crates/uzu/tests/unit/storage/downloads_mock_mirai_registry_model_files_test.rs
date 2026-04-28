use std::time::Duration;

use nagare::api::BaseUrl;
use shoji::traits::Registry;
use tokio::time::{sleep as tokio_sleep, timeout as tokio_timeout};
use uzu::{
    device::Device,
    registry::mirai::{Backend, Config as MiraiRegistryConfig, Registry as MiraiRegistry},
    storage::{Config as StorageConfig, Storage, types::DownloadPhase},
};

use crate::common::mock_mirai_server::MockMiraiServer;

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_downloads_files_from_mock_mirai_registry() {
    let server = MockMiraiServer::start().await;
    let device = Device::new().unwrap();
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
    .unwrap();
    let models = registry.models().await.unwrap();
    let model = models.first().cloned().unwrap();

    let temp_dir = tempfile::tempdir().unwrap();
    let storage_config = StorageConfig::new(device, Some(temp_dir.path().to_path_buf()), "mock_mirai_storage".to_string());
    let storage = Storage::new(tokio::runtime::Handle::current(), storage_config).await.unwrap();

    storage.refresh(models).await.unwrap();
    storage.download(&model.identifier).await.unwrap();

    let downloaded = tokio_timeout(Duration::from_secs(20), async {
        loop {
            let state = storage.state(&model.identifier).await.unwrap();
            if matches!(state.phase, DownloadPhase::Downloaded {}) {
                return state;
            }
            tokio_sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();

    assert_eq!(downloaded.downloaded_bytes, downloaded.total_bytes);
    let cache_path = storage.config.cache_model_path(&model).unwrap();
    for payload in server.payloads() {
        let file_path = cache_path.join(&payload.file.name);
        let downloaded_bytes =
            tokio::fs::read(&file_path).await.unwrap_or_else(|error| panic!("{}: {}", file_path.display(), error));
        assert_eq!(downloaded_bytes, payload.bytes.to_vec());
    }
}
