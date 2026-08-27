use download_manager::FileDownloadManagerType;
use kiban::rt::RuntimeHandle;
use shoji::types::model::Model;

use crate::{
    device::Device,
    models::{ModelsResolver, ResolvedModels},
    storage::{Config, Storage},
};

pub struct TestStorage {
    pub storage: Storage,
    _temp_dir_guard: tempfile::TempDir,
}

impl TestStorage {
    pub async fn with_models_and_manager(
        tokio_handle: RuntimeHandle,
        models: Vec<Model>,
        download_manager_type: FileDownloadManagerType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let temp_dir_guard = tempfile::tempdir()?;
        let base_path = temp_dir_guard.path().to_path_buf();
        let device = Device::new()?;
        let config = Config::new(device, Some(base_path.clone()), "test_storage".to_string())
            .with_download_manager_type(download_manager_type);
        let storage = Storage::new(tokio_handle, config).await?;
        let resolved = ModelsResolver::new(None, base_path.join("resolved-models.json"))?
            .resolve(models, &ResolvedModels::default())
            .await?;
        storage.refresh(&resolved, None).await?;
        Ok(Self {
            storage,
            _temp_dir_guard: temp_dir_guard,
        })
    }
}
