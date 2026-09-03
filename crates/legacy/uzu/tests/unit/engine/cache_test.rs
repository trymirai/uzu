use std::{error::Error, future::Future, pin::Pin};

use ::download_manager::FileDownloadManagerType;
use kiban::rt::RuntimeHandle;
use nagare::telemetry::Telemetry;
use shoji::{
    traits::Registry as RegistryTrait,
    types::model::{Model, ModelAccessibility},
};
use tempfile::tempdir;
use tokio::fs::read as tokio_read;

use super::*;
use crate::{
    models::{ModelCatalog, ModelsResolver, ResolvedModel, ResolvedModels},
    registry::FixedRegistry,
};

struct FailingRegistry(&'static str);

impl RegistryTrait for FailingRegistry {
    type Error = RegistryError;

    fn indentifier(&self) -> String {
        self.0.to_string()
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, RegistryError>> + Send + '_>> {
        Box::pin(async {
            Err(RegistryError::UnableToGetModels {
                message: "unavailable".to_string(),
            })
        })
    }
}

#[tokio::test]
async fn failed_refresh_preserves_snapshot_and_registry_changes_roll_back() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let cache_path = directory.path().join("resolved-models.json");
    let resolver = ModelsResolver::new(None, cache_path.clone())?;
    let model = Model::external(
        "cached-model".to_string(),
        "cached-registry".to_string(),
        "Cached Registry".to_string(),
        "backend".to_string(),
        "Backend".to_string(),
        "1".to_string(),
        Vec::new(),
        ModelAccessibility::Remote {
            repository: None,
        },
        None,
    );
    let previous = ResolvedModels::new(vec![ResolvedModel::passthrough(model)]);
    resolver.save_cache(&previous).await?;
    let cached = tokio_read(&cache_path).await?;
    let catalog = ModelCatalog::new(None, cache_path.clone()).await?;

    let mut registry = MergedRegistry::new(Vec::new());
    registry.add(Box::new(FailingRegistry("failing")))?;
    let runtime_handle = RuntimeHandle::try_current()?;
    let storage_config =
        StorageConfig::new(Device::new()?, Some(directory.path().to_path_buf()), "cache-test".to_string())
            .with_download_manager_type(FileDownloadManagerType::Universal);
    let engine = Engine {
        settings: SharedAccess::new(None),
        registry: SharedAccess::new(registry),
        catalog,
        catalog_refresh_lock: SharedAccess::new(()),
        storage: SharedAccess::new(Storage::new(runtime_handle, storage_config).await?),
        backends: SharedAccess::new(HashMap::new()),
        callback: SharedAccess::new(None),
        telemetry: SharedAccess::new(Telemetry::disabled()),
        huggingface_api_key: None,
    };

    engine.handle_initial_catalog_refresh().await?;
    let registries = engine.registry.lock().await.indentifier();

    assert!(engine.add_registry(Box::new(FixedRegistry::new("working".to_string(), Vec::new()))).await.is_err());
    assert_eq!(engine.registry.lock().await.indentifier(), registries);
    engine.registry.lock().await.add(Box::new(FailingRegistry("other")))?;
    let registries_with_other = engine.registry.lock().await.indentifier();
    assert!(engine.remove_registry("failing".to_string()).await.is_err());

    let identifiers = engine.models().await?.into_iter().map(|model| model.identifier).collect::<Vec<_>>();
    assert_eq!(identifiers, ["cached-model"]);
    assert_eq!(engine.registry.lock().await.indentifier(), registries_with_other);
    assert_eq!(tokio_read(cache_path).await?, cached);
    Ok(())
}
