use std::sync::Arc;

use kiban::rt::RuntimeHandle;
use shoji::types::model::Model;
use uzu::{
    device::Device,
    storage::{Config, DownloadManagerType, Storage},
};

pub struct TestStorage {
    pub storage: Arc<Storage>,
    _temp_dir_guard: tempfile::TempDir,
}

impl TestStorage {
    pub async fn new(
        tokio_handle: RuntimeHandle,
        models: Vec<Model>,
        download_manager_type: DownloadManagerType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let temp_dir_guard = tempfile::tempdir()?;
        let config = Config::new(
            Device::new()?,
            Some(temp_dir_guard.path().to_path_buf()),
            "test_storage".to_string(),
            download_manager_type,
        );
        let storage = Storage::new(tokio_handle, config).await?;
        storage.refresh(&models).await?;
        Ok(Self {
            storage: Arc::new(storage),
            _temp_dir_guard: temp_dir_guard,
        })
    }
}
