mod config;
mod downloader;
mod error;

use std::collections::HashMap;

use backend_uzu::inference_backend::{Backend as UzuBackend, Config as UzuBackendConfig};
pub use config::Config;
pub use downloader::Downloader;
pub use error::Error;
use shoji::{
    traits::{Backend, BackendInstance, Registry},
    types::Model,
};
use tokio::runtime::Handle;
use tokio_stream::wrappers::BroadcastStream;

use crate::{
    device::Device,
    helpers::SharedAccess,
    logs,
    registry::{
        CachedRegistry, Error as RegistryError, MergedRegistry,
        mirai::{Backend as MiraiBackend, Config as MiraiRegistryConfig, Registry as MiraiRegistry},
    },
    storage::{Config as StorageConfig, Storage, types::DownloadState},
};

pub struct Engine {
    registry: SharedAccess<MergedRegistry>,
    storage: SharedAccess<Storage>,
    backends: HashMap<String, Box<dyn Backend>>,
}

impl Engine {
    pub async fn new(config: Config) -> Result<Self, Error> {
        let tokio_handle = Handle::try_current().map_err(|error| Error::TokioError {
            message: error.to_string(),
        })?;

        let device = Device::new()?;
        let registry = SharedAccess::new(MergedRegistry::new(vec![]));
        let storage_config = StorageConfig::new(device.clone(), None, "mirai".to_string());
        logs::start(storage_config.cache_path(), &storage_config.log_name(), false);

        let storage = SharedAccess::new(Storage::new(tokio_handle.clone(), storage_config).await?);

        let uzu_backend_config = UzuBackendConfig {};
        let uzu_backend: Box<dyn Backend> =
            Box::new(UzuBackend::new(uzu_backend_config).map_err(|error| Error::Backend {
                message: error.to_string(),
            })?);

        let mirai_registry_config = MiraiRegistryConfig {
            api_key: config.mirai_api_key,
            device: device.clone(),
            backends: vec![MiraiBackend {
                identifier: uzu_backend.identifier(),
                version: uzu_backend.version(),
            }],
            include_traces: false,
        };
        let mirai_registry = Box::new(MiraiRegistry::new(mirai_registry_config)?);

        let mut engine = Self {
            storage,
            registry,
            backends: HashMap::new(),
        };
        engine.add_backend(uzu_backend);
        engine.add_registry(mirai_registry).await?;

        // if let Some(openai_api_key) = config.openai_api_key {
        //     let openai_registry = OpenAIRegistry::new(OpenAIConfig::openai(openai_api_key))?;
        //     engine.add_registry(Box::new(openai_registry)).await?;
        // }

        Ok(engine)
    }
}

impl Engine {
    pub async fn add_registry(
        &self,
        registry: Box<dyn Registry<Error = RegistryError>>,
    ) -> Result<(), Error> {
        self.registry.lock().await.add(Box::new(CachedRegistry::new(registry)))?;
        self.handle_registry_resfresh().await?;
        Ok(())
    }

    pub async fn remove_registry(
        &self,
        registry_identifier: &str,
    ) -> Result<(), Error> {
        self.registry.lock().await.remove(registry_identifier)?;
        self.handle_registry_resfresh().await?;
        Ok(())
    }
}

impl Engine {
    pub fn add_backend(
        &mut self,
        backend: Box<dyn Backend>,
    ) {
        self.backends.insert(backend.identifier(), backend);
    }

    pub fn remove_backend(
        &mut self,
        identifier: &str,
    ) {
        self.backends.remove(identifier);
    }
}

impl Engine {
    pub async fn models(&self) -> Result<Vec<Model>, Error> {
        self.registry.lock().await.models().await.map_err(Error::from)
    }

    pub async fn model_by_identifier(
        &self,
        identifier: &str,
    ) -> Result<Option<Model>, Error> {
        self.registry.lock().await.model_by_identifier(identifier).await.map_err(Error::from)
    }

    pub async fn model_by_repo_id(
        &self,
        repo_id: &str,
    ) -> Result<Option<Model>, Error> {
        self.registry.lock().await.model_by_repo_id(repo_id).await.map_err(Error::from)
    }
}

impl Engine {
    pub fn downloader(
        &self,
        model: &Model,
    ) -> Downloader {
        Downloader::new(model.identifier(), self.storage.clone())
    }

    pub async fn storage_subscribe(&self) -> BroadcastStream<(String, DownloadState)> {
        self.storage.lock().await.subscribe()
    }

    async fn handle_registry_resfresh(&self) -> Result<(), Error> {
        let models = self.registry.lock().await.models().await?;
        self.storage.lock().await.refresh(models).await?;
        Ok(())
    }
}
