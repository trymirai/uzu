#![allow(dead_code)]

use std::path::PathBuf;

use download_manager::FileDownloadManagerType;
use kiban::rt::RuntimeHandle;
use shoji::types::model::Model;
use uzu::{
    device::Device,
    registry::FixedRegistry,
    storage::{Config, Storage},
};

pub struct TestStorage {
    pub config: Config,
    pub registry: FixedRegistry,
    pub storage: Storage,
    pub base_path: PathBuf,
    _temp_dir_guard: tempfile::TempDir,
}

impl TestStorage {
    pub async fn with_models(
        tokio_handle: RuntimeHandle,
        models: Vec<Model>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_models_and_manager(tokio_handle, models, FileDownloadManagerType::default()).await
    }

    pub async fn with_models_and_manager(
        tokio_handle: RuntimeHandle,
        models: Vec<Model>,
        download_manager_type: FileDownloadManagerType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let temp_dir_guard = tempfile::tempdir()?;
        let base_path = temp_dir_guard.path().to_path_buf();
        let registry = FixedRegistry::new("test_registry".to_string(), models.clone());
        let device = Device::new()?;
        let config = Config::new(device, Some(base_path.clone()), "test_storage".to_string())
            .with_download_manager_type(download_manager_type);
        let storage = Storage::new(tokio_handle, config.clone()).await?;
        storage.refresh(models).await?;
        Ok(Self {
            config,
            registry,
            storage,
            base_path,
            _temp_dir_guard: temp_dir_guard,
        })
    }
}
