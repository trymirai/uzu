#![allow(dead_code)]

use download_manager::FileDownloadManagerType;
use mock_registry::{Behavior, MockRegistry};
use shoji::types::model::Model;
use tokio::runtime::Handle as TokioHandle;
use uzu::{
    device::Device,
    engine::Downloader,
    helpers::SharedAccess,
    storage::{Config, Storage},
};

pub struct TestEngineFixture {
    pub registry: MockRegistry,
    pub model: Model,
    pub storage: SharedAccess<Storage>,
    pub downloader: Downloader,
    _temp_dir_guard: tempfile::TempDir,
}

impl TestEngineFixture {
    pub async fn start(download_manager_type: FileDownloadManagerType) -> Result<Self, Box<dyn std::error::Error>> {
        Self::start_with(download_manager_type, Behavior::empty()).await
    }

    pub async fn start_with(
        download_manager_type: FileDownloadManagerType,
        behavior: Behavior,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let registry = MockRegistry::start_with(behavior).await?;
        let model = registry.models.first().cloned().ok_or("mock registry must include a model")?;

        let temp_dir_guard = tempfile::tempdir()?;
        let device = Device::new()?;
        let config = Config::new(device, Some(temp_dir_guard.path().to_path_buf()), "test_storage".to_string())
            .with_download_manager_type(download_manager_type);
        let storage = Storage::new(TokioHandle::current(), config).await?;
        storage.refresh(vec![model.clone()]).await?;

        let storage_shared = SharedAccess::new(storage);
        let downloader = Downloader::new(model.identifier.clone(), storage_shared.clone());

        Ok(Self {
            registry,
            model,
            storage: storage_shared,
            downloader,
            _temp_dir_guard: temp_dir_guard,
        })
    }
}
