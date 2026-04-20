mod config;
mod downloader;
mod error;

use std::{collections::HashMap, sync::Arc};

use backend_remote::openai::Backend as OpenAIBackend;
use backend_uzu::inference_backend::Backend as UzuBackend;
pub use config::Config;
pub use downloader::Downloader;
pub use error::Error;
use nagare::chat::Session as ChatSession;
use shoji::{
    traits::{Backend, Registry},
    types::model::Model,
};
use tokio::runtime::Handle;
use tokio_stream::wrappers::BroadcastStream;

use crate::{
    device::Device,
    helpers::{SharedAccess, is_endpoint_reachable},
    logs,
    registry::{
        CachedRegistry, Error as RegistryError, MergedRegistry,
        local::{Config as LocalRegistryConfig, Registry as LocalRegistry},
        mirai::{Backend as MiraiBackend, Config as MiraiRegistryConfig, Registry as MiraiRegistry},
        openai::{Config as OpenAIConfig, Registry as OpenAIRegistry},
    },
    storage::{
        Config as StorageConfig, Storage,
        types::{DownloadPhase, DownloadState},
    },
};

pub struct Engine {
    registry: SharedAccess<MergedRegistry>,
    storage: SharedAccess<Storage>,
    backends: HashMap<String, Arc<dyn Backend>>,
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

        let mut engine = Self {
            storage,
            registry,
            backends: HashMap::new(),
        };

        {
            let uzu_backend = UzuBackend::new().map_err(|_| Error::UnableToCreateBackend)?;
            let uzu_backend_identifier = uzu_backend.identifier();
            let uzu_backend_version = uzu_backend.version();

            let mirai_registry_config = MiraiRegistryConfig {
                api_key: config.mirai_api_key,
                device: device.clone(),
                backends: vec![MiraiBackend {
                    identifier: uzu_backend_identifier.clone(),
                    version: uzu_backend_version,
                }],
                include_traces: false,
            };
            let mirai_registry = Box::new(MiraiRegistry::new(mirai_registry_config)?);

            engine.add_backend(Arc::new(uzu_backend) as Arc<dyn Backend>);
            engine.add_registry(mirai_registry).await?;

            if let Some(lalamo_path) = config.lalamo_path {
                let lalamo_registry =
                    LocalRegistry::new(LocalRegistryConfig::lalamo(uzu_backend_identifier, lalamo_path))?;
                engine.add_registry(Box::new(lalamo_registry)).await?;
            }
        }

        let mut openai_configs: Vec<OpenAIConfig> = vec![];
        {
            let ollama_config = OpenAIConfig::ollama();
            if is_endpoint_reachable(&ollama_config.api_endpoint).await {
                openai_configs.push(ollama_config);
            }
        }
        {
            let lmstudio_config = OpenAIConfig::lmstudio();
            if is_endpoint_reachable(&lmstudio_config.api_endpoint).await {
                openai_configs.push(lmstudio_config);
            }
        }
        if let Some(openai_api_key) = config.openai_api_key {
            openai_configs.push(OpenAIConfig::openai(openai_api_key));
        }
        if let Some(anthropic_api_key) = config.anthropic_api_key {
            openai_configs.push(OpenAIConfig::anthropic(anthropic_api_key));
        }
        if let Some(gemini_api_key) = config.gemini_api_key {
            openai_configs.push(OpenAIConfig::gemini(gemini_api_key));
        }
        if let Some(xai_api_key) = config.xai_api_key {
            openai_configs.push(OpenAIConfig::xai(xai_api_key));
        }
        if let Some(baseten_api_key) = config.baseten_api_key {
            openai_configs.push(OpenAIConfig::baseten(baseten_api_key));
        }
        if let Some(openrouter_api_key) = config.openrouter_api_key {
            openai_configs.push(OpenAIConfig::openrouter(openrouter_api_key));
        }
        for config in openai_configs {
            let registry = OpenAIRegistry::new(config.clone())?;
            let backend = OpenAIBackend::new(config.into()).map_err(|_| Error::UnableToCreateBackend)?;
            engine.add_registry(Box::new(registry)).await?;
            engine.add_backend(Arc::new(backend) as Arc<dyn Backend>);
        }

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
        backend: Arc<dyn Backend>,
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

    pub async fn model(
        &self,
        identifier: &str,
    ) -> Result<Option<Model>, Error> {
        if let Some(model) = self.model_by_identifier(identifier).await? {
            return Ok(Some(model));
        }
        if let Some(model) = self.model_by_repo_id(identifier).await? {
            return Ok(Some(model));
        }
        self.model_by_path(identifier).await
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

    pub async fn model_by_path(
        &self,
        path: &str,
    ) -> Result<Option<Model>, Error> {
        let models = self.models().await?;
        for model in models {
            if let Some(model_path) = self.model_path(&model).await {
                if model_path == path {
                    return Ok(Some(model));
                }
            }
        }
        Ok(None)
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

    pub async fn model_path(
        &self,
        model: &Model,
    ) -> Option<String> {
        let path = if model.is_local() {
            if let Some(local_external_path) = model.local_external_path() {
                Some(local_external_path.clone())
            } else {
                let storage = self.storage.lock().await;
                let state = storage.state(&model.identifier()).await?;
                match state.phase {
                    DownloadPhase::Downloaded => {
                        storage.config.cache_model_path(model).map(|path| path.to_string_lossy().to_string())
                    },
                    DownloadPhase::NotDownloaded
                    | DownloadPhase::Downloading
                    | DownloadPhase::Paused
                    | DownloadPhase::Locked
                    | DownloadPhase::Error(_) => None,
                }
            }
        } else {
            None
        };
        path
    }

    async fn handle_registry_resfresh(&self) -> Result<(), Error> {
        let models = self.registry.lock().await.models().await?;
        self.storage.lock().await.refresh(models).await?;
        Ok(())
    }
}

impl Engine {
    pub async fn chat(
        &self,
        model: Model,
    ) -> Result<ChatSession, Error> {
        let path = self.model_path(&model).await;
        if let Some(backend_entity) = model.backend_entity() {
            let backend = self.backends.get(&backend_entity.identifier).ok_or(Error::BackendNotFound)?;
            let session = ChatSession::new(backend.as_ref(), model, path).await?;
            Ok(session)
        } else {
            return Err(Error::BackendNotFound);
        }
    }
}
