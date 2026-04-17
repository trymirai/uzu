mod config;
mod downloader;
mod error;

pub use config::Config;
pub use downloader::Downloader;
pub use error::Error;
use nagare::{
    device::Device,
    helpers::SharedAccess,
    registry::{CachedRegistry, MergedRegistry, Registry, types::Model},
    storage::{Config as StorageConfig, Storage},
};
use tokio::runtime::Handle;

pub struct Engine {
    registry: SharedAccess<MergedRegistry>,
    storage: SharedAccess<Storage>,
}

impl Engine {
    pub async fn new(_config: Config) -> Result<Self, Error> {
        let tokio_handle = Handle::try_current().map_err(|error| Error::TokioError {
            message: error.to_string(),
        })?;

        let device = Device::new()?;
        let registry = SharedAccess::new(MergedRegistry::new(vec![]));
        let storage = SharedAccess::new(
            Storage::new(tokio_handle.clone(), StorageConfig::new(device.clone(), None, "mirai".to_string())).await?,
        );

        let engine = Self {
            storage,
            registry,
        };

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
        registry: Box<dyn Registry>,
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

    async fn handle_registry_resfresh(&self) -> Result<(), Error> {
        let models = self.registry.lock().await.models().await?;
        self.storage.lock().await.refresh(models).await?;
        Ok(())
    }
}
