mod callback;
mod config;
mod downloader;
mod error;

use std::{collections::HashMap, sync::Arc};

use backend_remote::openai::Backend as OpenAIBackend;
use backend_uzu::inference::Backend as UzuBackend;
pub use callback::{EngineCallback, EngineCallbackType};
pub use config::EngineConfig;
pub use downloader::{Downloader, DownloaderStream, DownloaderStreamUpdate};
pub use error::EngineError;
use nagare::{chat::ChatSession, classification::ClassificationSession, text_to_speech::TextToSpeechSession};
use shoji::{
    traits::{Backend, Registry},
    types::{model::Model, session::chat::ChatConfig},
};
use tokio::runtime::Handle;
use tokio_stream::wrappers::BroadcastStream;

use crate::{
    device::Device,
    helpers::{SharedAccess, is_endpoint_reachable},
    logs,
    registry::{
        CachedRegistry, MergedRegistry, RegistryError,
        local::{Config as LocalRegistryConfig, Registry as LocalRegistry},
        mirai::{Backend as MiraiBackend, Config as MiraiRegistryConfig, Registry as MiraiRegistry},
        openai::{Config as OpenAIConfig, Registry as OpenAIRegistry},
    },
    storage::{
        Config as StorageConfig, Storage,
        types::{DownloadPhase, DownloadState},
    },
};

#[bindings::export(Class)]
pub struct Engine {
    registry: SharedAccess<MergedRegistry>,
    storage: SharedAccess<Storage>,
    backends: SharedAccess<HashMap<String, Arc<dyn Backend>>>,
    callback: SharedAccess<Option<Arc<EngineCallback>>>,
}

impl Engine {
    pub async fn new(config: EngineConfig) -> Result<Self, EngineError> {
        let tokio_handle = Handle::try_current().map_err(|error| EngineError::TokioError {
            message: error.to_string(),
        })?;

        let device = Device::new()?;
        let registry = SharedAccess::new(MergedRegistry::new(vec![]));
        let storage_config = StorageConfig::new(device.clone(), None, "mirai".to_string());
        logs::start(storage_config.cache_path(), &storage_config.log_name(), false);

        let storage = SharedAccess::new(Storage::new(tokio_handle.clone(), storage_config).await?);

        let engine = Self {
            storage,
            registry,
            backends: SharedAccess::new(HashMap::new()),
            callback: SharedAccess::new(None),
        };

        {
            let uzu_backend = UzuBackend::new();
            let uzu_backend_identifier = uzu_backend.identifier();
            let uzu_backend_version = uzu_backend.version();

            let mirai_registry_config = MiraiRegistryConfig {
                api_key: config.mirai_api_key,
                device: device.clone(),
                backends: vec![MiraiBackend {
                    identifier: uzu_backend_identifier.clone(),
                    version: uzu_backend_version.clone(),
                }],
                include_traces: false,
            };
            let mirai_registry = Box::new(MiraiRegistry::new(mirai_registry_config)?);

            engine.add_backend(Arc::new(uzu_backend) as Arc<dyn Backend>).await;
            engine.add_registry(mirai_registry).await?;

            if let Some(lalamo_path) = config.lalamo_path {
                let lalamo_registry = LocalRegistry::new(LocalRegistryConfig::lalamo(
                    uzu_backend_identifier,
                    uzu_backend_version,
                    lalamo_path,
                ))?;
                engine.add_registry(Box::new(lalamo_registry)).await?;
            }
        }

        let mut openai_configs: Vec<OpenAIConfig> = vec![];
        if config.allow_ollama_usage {
            let ollama_config = OpenAIConfig::ollama();
            if is_endpoint_reachable(&ollama_config.api_endpoint).await {
                openai_configs.push(ollama_config);
            }
        }
        if config.allow_lmstudio_usage {
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
            let backend = OpenAIBackend::new(config.into()).map_err(|_| EngineError::UnableToCreateBackend {})?;
            engine.add_registry(Box::new(registry)).await?;
            engine.add_backend(Arc::new(backend) as Arc<dyn Backend>).await;
        }

        Ok(engine)
    }
}

#[bindings::export(Implementation)]
impl Engine {
    #[bindings::export(Factory)]
    pub async fn create(config: EngineConfig) -> Result<Self, EngineError> {
        Self::new(config).await
    }
}

#[bindings::export(Implementation)]
impl Engine {
    #[bindings::export(Method)]
    pub async fn register_callback(
        &self,
        callback: &EngineCallback,
    ) -> Result<(), EngineError> {
        self.callback.lock().await.replace(Arc::new(callback.clone()));
        Ok(())
    }
}

impl Engine {
    pub async fn add_registry(
        &self,
        registry: Box<dyn Registry<Error = RegistryError>>,
    ) -> Result<(), EngineError> {
        self.registry.lock().await.add(Box::new(CachedRegistry::new(registry)))?;
        self.handle_registry_resfresh().await?;
        Ok(())
    }

    pub async fn add_backend(
        &self,
        backend: Arc<dyn Backend>,
    ) {
        self.backends.lock().await.insert(backend.identifier(), backend);
    }
}

#[bindings::export(Implementation)]
impl Engine {
    #[bindings::export(Method)]
    pub async fn remove_registry(
        &self,
        registry_identifier: String,
    ) -> Result<(), EngineError> {
        self.registry.lock().await.remove(&registry_identifier)?;
        self.handle_registry_resfresh().await?;
        Ok(())
    }

    #[bindings::export(Method)]
    pub async fn remove_backend(
        &self,
        identifier: String,
    ) {
        self.backends.lock().await.remove(&identifier);
    }
}

#[bindings::export(Implementation)]
impl Engine {
    #[bindings::export(Getter)]
    pub async fn models(&self) -> Result<Vec<Model>, EngineError> {
        self.registry.lock().await.models().await.map_err(EngineError::from)
    }

    #[bindings::export(Getter)]
    pub async fn models_local(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_local()).collect())
    }

    #[bindings::export(Getter)]
    pub async fn models_remote(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_remote()).collect())
    }

    #[bindings::export(Getter)]
    pub async fn models_downloadable(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_downloadable()).collect())
    }

    #[bindings::export(Getter)]
    pub async fn models_for_chat(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_chat_capable()).collect())
    }

    #[bindings::export(Getter)]
    pub async fn models_for_classification(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_classification_capable()).collect())
    }

    #[bindings::export(Getter)]
    pub async fn models_for_text_to_speech(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_text_to_speech_capable()).collect())
    }

    #[bindings::export(Getter)]
    pub async fn models_for_translation(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_translation_capable()).collect())
    }

    #[bindings::export(Getter)]
    pub async fn models_for_speculation(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_speculation_capable()).collect())
    }

    #[bindings::export(Method)]
    pub async fn models_by_vendor(
        &self,
        vendor_identifier: String,
    ) -> Result<Vec<Model>, EngineError> {
        Ok(self
            .models()
            .await?
            .into_iter()
            .filter(|model| {
                model.family.as_ref().map(|family| family.vendor.identifier == vendor_identifier).unwrap_or(false)
            })
            .collect())
    }

    #[bindings::export(Method)]
    pub async fn models_by_family(
        &self,
        family_identifier: String,
    ) -> Result<Vec<Model>, EngineError> {
        Ok(self
            .models()
            .await?
            .into_iter()
            .filter(|model| model.family.as_ref().map(|family| family.identifier == family_identifier).unwrap_or(false))
            .collect())
    }
}

#[bindings::export(Implementation)]
impl Engine {
    #[bindings::export(Method)]
    pub async fn model(
        &self,
        identifier: String,
    ) -> Result<Option<Model>, EngineError> {
        if let Some(model) = self.model_by_identifier(identifier.clone()).await? {
            return Ok(Some(model));
        }
        if let Some(model) = self.model_by_repo_id(identifier.clone()).await? {
            return Ok(Some(model));
        }
        self.model_by_path(identifier.clone()).await
    }

    #[bindings::export(Method)]
    pub async fn model_by_identifier(
        &self,
        identifier: String,
    ) -> Result<Option<Model>, EngineError> {
        self.registry.lock().await.model_by_identifier(&identifier).await.map_err(EngineError::from)
    }

    #[bindings::export(Method)]
    pub async fn model_by_repo_id(
        &self,
        repo_id: String,
    ) -> Result<Option<Model>, EngineError> {
        self.registry.lock().await.model_by_repo_id(&repo_id).await.map_err(EngineError::from)
    }

    #[bindings::export(Method)]
    pub async fn model_by_path(
        &self,
        path: String,
    ) -> Result<Option<Model>, EngineError> {
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

#[bindings::export(Implementation)]
impl Engine {
    #[bindings::export(Method)]
    pub async fn model_path(
        &self,
        model: &Model,
    ) -> Option<String> {
        let path = if model.is_local() {
            if let Some(local_external_path) = model.local_external_path() {
                Some(local_external_path.clone())
            } else {
                let storage = self.storage.lock().await;
                let state = storage.state(&model.identifier.clone()).await?;
                match state.phase {
                    DownloadPhase::Downloaded {} => {
                        storage.config.cache_model_path(model).map(|path| path.to_string_lossy().to_string())
                    },
                    DownloadPhase::NotDownloaded {}
                    | DownloadPhase::Downloading {}
                    | DownloadPhase::Paused {}
                    | DownloadPhase::Locked {}
                    | DownloadPhase::Error {
                        ..
                    } => None,
                }
            }
        } else {
            None
        };
        path
    }

    #[bindings::export(Method)]
    pub fn downloader(
        &self,
        model: &Model,
    ) -> Downloader {
        Downloader::new(model.identifier.clone(), self.storage.clone())
    }

    #[bindings::export(Method)]
    pub async fn download(
        &self,
        model: &Model,
    ) -> Result<DownloaderStream, EngineError> {
        if !model.is_downloadable() {
            return Ok(DownloaderStream::empty(model.identifier.clone()));
        }

        let downloader = self.downloader(model);
        let Some(state) = downloader.state().await else {
            return Err(EngineError::UnableToGetDownloaderProgressStream {});
        };
        if matches!(state.phase, DownloadPhase::Downloaded {}) {
            return Ok(DownloaderStream::empty(model.identifier.clone()));
        }
        downloader.resume().await?;
        Ok(downloader.progress().await?)
    }

    #[bindings::export(Method)]
    pub async fn download_state(
        &self,
        model: &Model,
    ) -> Option<DownloadState> {
        self.downloader(model).state().await
    }
}

#[bindings::export(Implementation)]
impl Engine {
    #[bindings::export(Method)]
    pub async fn chat(
        &self,
        model: Model,
        config: ChatConfig,
    ) -> Result<ChatSession, EngineError> {
        let path = self.model_path(&model).await;
        if let Some(backend) = model.backends.first() {
            let backends = self.backends.lock().await;
            let backend = backends.get(&backend.identifier).ok_or(EngineError::BackendNotFound {})?;
            let session = ChatSession::new(backend.as_ref(), config, model, path).await?;
            Ok(session)
        } else {
            return Err(EngineError::BackendNotFound {});
        }
    }

    #[bindings::export(Method)]
    pub async fn classification(
        &self,
        model: Model,
    ) -> Result<ClassificationSession, EngineError> {
        let path = self.model_path(&model).await;
        if let Some(backend) = model.backends.first() {
            let backends = self.backends.lock().await;
            let backend = backends.get(&backend.identifier).ok_or(EngineError::BackendNotFound {})?;
            let session = ClassificationSession::new(backend.as_ref(), model, path).await?;
            Ok(session)
        } else {
            return Err(EngineError::BackendNotFound {});
        }
    }

    #[bindings::export(Method)]
    pub async fn text_to_speech(
        &self,
        model: Model,
    ) -> Result<TextToSpeechSession, EngineError> {
        let path = self.model_path(&model).await;
        if let Some(backend) = model.backends.first() {
            let backends = self.backends.lock().await;
            let backend = backends.get(&backend.identifier).ok_or(EngineError::BackendNotFound {})?;
            let session = TextToSpeechSession::new(backend.as_ref(), model, path).await?;
            Ok(session)
        } else {
            return Err(EngineError::BackendNotFound {});
        }
    }
}

impl Engine {
    pub async fn storage_subscribe(&self) -> BroadcastStream<(String, DownloadState)> {
        self.storage.lock().await.subscribe()
    }

    async fn handle_registry_resfresh(&self) -> Result<(), EngineError> {
        let models = self.registry.lock().await.models().await?;
        self.storage.lock().await.refresh(models).await?;
        Ok(())
    }
}
