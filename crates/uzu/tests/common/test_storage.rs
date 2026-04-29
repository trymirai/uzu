#![allow(dead_code)]

use std::path::PathBuf;

use shoji::types::model::Model;
use tokio::runtime::Handle as TokioHandle;
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
        tokio_handle: TokioHandle,
        models: Vec<Model>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let temp_dir_guard = tempfile::tempdir()?;
        let base_path = temp_dir_guard.path().to_path_buf();
        let registry = FixedRegistry::new("test_registry".to_string(), models.clone());
        let device = Device::new()?;
        let config = Config::new(device, Some(base_path.clone()), "test_storage".to_string());
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
