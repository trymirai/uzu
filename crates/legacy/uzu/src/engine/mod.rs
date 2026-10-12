pub mod bridge;
mod callback;
#[cfg(test)]
mod catalog_tests;
pub mod config;
mod downloader;
mod downloader_stream;
mod error;
mod sampling;
mod shorthand;

use std::{
    collections::HashMap,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use backend_remote::openai::Backend as OpenAIBackend;
pub use callback::{EngineCallback, EngineCallbackType};
pub use config::EngineConfig;
pub use downloader::Downloader;
pub use downloader_stream::DownloaderStream;
pub use error::EngineError;
use futures_util::stream::FuturesUnordered;
use indexmap::IndexSet;
use kiban::rt::RuntimeHandle;
use nagare::{
    chat::{ChatInstance, ChatSession},
    classification::ClassificationSession,
    text_to_speech::TextToSpeechSession,
};
use shoji::{
    traits::{Backend, Registry},
    types::{
        model::{Model, ModelFamily, ModelIdentifier, ModelRegistry, ModelVendor},
        session::chat::ChatConfig,
    },
};
use sysinfo::System;
use tokio::sync::{broadcast, oneshot, watch};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

use crate::{
    device::Device,
    engine::bridge::UzuLlmBackend,
    helpers::{SharedAccess, is_endpoint_reachable},
    logs,
    registry::{
        CachedRegistry, MergedRegistry, RegistryError,
        local::{Config as LocalRegistryConfig, Registry as LocalRegistry},
        mirai::{Backend as MiraiBackend, HUGGING_FACE_URL, Registry as MiraiRegistry},
        openai::{Config as OpenAIConfig, Registry as OpenAIRegistry},
        unique_model,
    },
    settings::Settings,
    storage::{BearerToken, Config as StorageConfig, DownloadPhase, DownloadState, Storage},
};

#[bindings::export(Class)]
#[derive(Clone)]
pub struct Engine {
    settings: SharedAccess<Option<Settings>>,
    registry: SharedAccess<MergedRegistry>,
    storage: Arc<Storage>,
    backends: SharedAccess<HashMap<String, Arc<dyn Backend>>>,
    callback: SharedAccess<Option<Arc<EngineCallback>>>,
    catalog_events: broadcast::Sender<()>,
    catalog_refreshes: Arc<AtomicUsize>,
    catalog_publish: Arc<tokio::sync::Mutex<()>>,
    published_catalog: Arc<tokio::sync::Mutex<(Vec<Model>, bool)>>,
}

impl Engine {
    pub async fn new(config: EngineConfig) -> Result<Self, EngineError> {
        let runtime_handle = RuntimeHandle::try_current().map_err(|error| EngineError::TokioError {
            message: error.to_string(),
        })?;

        let settings = if let Some(application_identifier) = &config.application_identifier {
            Some(Settings::new(application_identifier.clone())?)
        } else {
            None
        };
        let mut config = config;
        if let Some(settings) = &settings {
            config.synchronize_with_settings(settings)?;
        }

        let device = Device::new()?;

        let registry = SharedAccess::new(MergedRegistry::new(vec![]));
        let huggingface_api_key = config.huggingface_api_key.map(BearerToken::from);
        let storage_config = StorageConfig::new(
            device.clone(),
            None,
            "mirai".to_string(),
            config.download_manager_type,
            HUGGING_FACE_URL.to_string(),
            huggingface_api_key.clone(),
        );
        let storage_cache_path = Storage::cache_path(&storage_config);
        logs::start(storage_cache_path.clone(), &format!("{}.log", storage_config.name), false);
        let storage = Arc::new(Storage::new(runtime_handle, storage_config).await?);

        let engine = Self {
            settings: SharedAccess::new(settings),
            storage,
            registry,
            backends: SharedAccess::new(HashMap::new()),
            callback: SharedAccess::new(None),
            catalog_events: broadcast::channel(64).0,
            catalog_refreshes: Arc::new(AtomicUsize::new(0)),
            catalog_publish: Arc::new(tokio::sync::Mutex::new(())),
            published_catalog: Arc::new(tokio::sync::Mutex::new((Vec::new(), true))),
        };
        engine.spawn_storage_listener();
        let mut initializations = FuturesUnordered::new();

        {
            let uzu_backend = UzuLlmBackend::new();
            let uzu_backend_identifier = uzu_backend.identifier();
            let uzu_backend_version = uzu_backend.version();

            let mirai_registry = Box::new(
                MiraiRegistry::builder()
                    .maybe_api_key(config.mirai_api_key)
                    .maybe_huggingface_api_key(huggingface_api_key)
                    .device(device.clone())
                    .backends(vec![MiraiBackend {
                        identifier: uzu_backend_identifier.clone(),
                        version: uzu_backend_version.clone(),
                    }])
                    .cache_path(storage_cache_path)
                    .build()?,
            );

            engine.add_backend(Arc::new(uzu_backend) as Arc<dyn Backend>).await;
            initializations.push(engine.start_registry(mirai_registry).await?);

            if let Some(local_path) = config.local_path {
                let local_registry = LocalRegistry::new(LocalRegistryConfig::new(
                    uzu_backend_identifier.clone(),
                    uzu_backend_version.clone(),
                    local_path,
                ))?;
                initializations.push(engine.start_registry(Box::new(local_registry)).await?);
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
            initializations.push(engine.start_registry(Box::new(registry)).await?);
            engine.add_backend(Arc::new(backend) as Arc<dyn Backend>).await;
        }

        let mut last_error = None;
        while engine.catalog_snapshot().await?.0.is_empty() {
            let Some(result) = initializations.next().await else {
                if let Some(error) = last_error {
                    return Err(error);
                }
                break;
            };
            if let Err(error) = result
                .map_err(|error| EngineError::TokioError {
                    message: error.to_string(),
                })
                .and_then(|result| result)
            {
                last_error = Some(error);
            }
        }
        Ok(engine)
    }

    pub fn toolchain_version() -> String {
        uzu_engine::TOOLCHAIN_VERSION.to_string()
    }

    pub fn version() -> String {
        uzu_engine::VERSION.to_string()
    }
}

#[bindings::export(Implementation)]
impl Engine {
    #[bindings::export(Method(Factory))]
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
        let first_update = self.start_registry(registry).await?;
        if self.catalog_snapshot().await?.0.is_empty() {
            first_update.await.map_err(|error| EngineError::TokioError {
                message: error.to_string(),
            })??;
        }
        Ok(())
    }

    async fn start_registry(
        &self,
        registry: Box<dyn Registry<Error = RegistryError>>,
    ) -> Result<oneshot::Receiver<Result<(), EngineError>>, EngineError> {
        let registry: Arc<dyn Registry<Error = RegistryError>> = Arc::new(CachedRegistry::new(registry));
        self.registry.lock().await.add(Arc::clone(&registry))?;
        self.start_registry_refresh(registry).await
    }

    async fn start_registry_refresh(
        &self,
        registry: Arc<dyn Registry<Error = RegistryError>>,
    ) -> Result<oneshot::Receiver<Result<(), EngineError>>, EngineError> {
        self.catalog_refreshes.fetch_add(1, Ordering::SeqCst);
        self.handle_registry_refresh(false).await?;
        let (ready, first_update) = oneshot::channel();
        let engine = self.clone();
        kiban::rt::spawn(async move {
            let (updates, mut updated) = watch::channel(());
            let publisher = updates.clone();
            let on_update: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                publisher.send_replace(());
            });
            let refresh = registry.refresh_listing(on_update);
            tokio::pin!(refresh);
            let mut ready = Some(ready);
            loop {
                tokio::select! {
                    result = &mut refresh => {
                        let published = engine.handle_registry_refresh(true).await;
                        if let Err(error) = &result {
                            tracing::warn!(%error, registry = %registry.identifier(), "catalog refresh failed");
                        }
                        if let Some(ready) = ready.take() {
                            let _ = ready.send(result.map(|_| ()).map_err(EngineError::from).and(published));
                        }
                        break;
                    },
                    update = updated.changed() => {
                        if update.is_err() { break; }
                        if let Err(error) = engine.handle_registry_refresh(false).await {
                            tracing::warn!(%error, "unable to publish catalog update");
                        }
                        if ready.is_some() && engine.catalog_snapshot().await.is_ok_and(|(models, _)| !models.is_empty()) {
                            let _ = ready.take().unwrap().send(Ok(()));
                        }
                    },
                }
            }
        });
        Ok(first_update)
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
        self.handle_registry_refresh(false).await?;
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
    #[bindings::export(Method(Getter))]
    pub async fn models(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.catalog_snapshot().await?.0)
    }

    #[bindings::export(Method(Getter))]
    pub async fn models_on_device(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_on_device()).collect())
    }

    #[bindings::export(Method(Getter))]
    pub async fn models_remote(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_remote()).collect())
    }

    #[bindings::export(Method(Getter))]
    pub async fn models_downloadable(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_downloadable()).collect())
    }

    #[bindings::export(Method(Getter))]
    pub async fn models_for_chat(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_chat_capable()).collect())
    }

    #[bindings::export(Method(Getter))]
    pub async fn models_for_classification(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_classification_capable()).collect())
    }

    #[bindings::export(Method(Getter))]
    pub async fn models_for_text_to_speech(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_text_to_speech_capable()).collect())
    }

    #[bindings::export(Method(Getter))]
    pub async fn models_for_translation(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_translation_capable()).collect())
    }

    #[bindings::export(Method(Getter))]
    pub async fn models_for_speculation(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_speculation_capable()).collect())
    }

    #[bindings::export(Method(Getter))]
    pub async fn model_registries(&self) -> Result<Vec<ModelRegistry>, EngineError> {
        let registries: Vec<_> = self
            .models()
            .await?
            .into_iter()
            .map(|model| model.registry.clone())
            .collect::<IndexSet<_>>()
            .into_iter()
            .collect();
        Ok(registries)
    }

    #[bindings::export(Method(Getter))]
    pub async fn model_vendors(&self) -> Result<Vec<ModelVendor>, EngineError> {
        let vendors: Vec<_> = self
            .model_families()
            .await?
            .into_iter()
            .map(|family| family.vendor.clone())
            .collect::<IndexSet<_>>()
            .into_iter()
            .collect();
        Ok(vendors)
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

    #[bindings::export(Method(Getter))]
    pub async fn model_families(&self) -> Result<Vec<ModelFamily>, EngineError> {
        let families: Vec<_> = self
            .models()
            .await?
            .into_iter()
            .filter_map(|model| model.family.clone())
            .collect::<IndexSet<_>>()
            .into_iter()
            .collect();
        Ok(families)
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

impl Engine {
    async fn find_catalog_model(
        &self,
        reference: &str,
        matches: impl Fn(&Model) -> bool,
    ) -> Result<Option<Model>, EngineError> {
        let mut updates = self.catalog_subscribe();
        loop {
            // Observe completion before taking the snapshot so a refresh that
            // finishes during the read cannot make a partial snapshot final.
            let refreshing = self.catalog_is_refreshing();
            let model = unique_model(reference, self.models().await?.into_iter().filter(&matches))?;
            if model.is_some() || !refreshing {
                return Ok(model);
            }
            if updates.next().await.is_none() {
                return Ok(None);
            }
        }
    }
}

#[bindings::export(Implementation)]
impl Engine {
    #[bindings::export(Method)]
    pub async fn model(
        &self,
        identifier: String,
    ) -> Result<Option<Model>, EngineError> {
        let is_directory = Path::new(&identifier).is_dir();
        let matches = |model: &Model| model.identifier == identifier || model.repo_ids().contains(&identifier);
        let registered = if is_directory {
            unique_model(&identifier, self.models().await?.into_iter().filter(matches))?
        } else {
            self.find_catalog_model(&identifier, matches).await?
        };
        if registered.is_some() && is_directory {
            return Err(RegistryError::UnableToGetModels {
                message: format!(
                    "Ambiguous model reference `{identifier}`: matches both a registered model and a directory"
                ),
            }
            .into());
        }
        let by_path = self.model_by_path(identifier.clone()).await?;
        if let Some(model) = unique_model(&identifier, registered.into_iter().chain(by_path))? {
            return Ok(Some(model));
        }

        let models = self.models().await?;
        let mut system = System::new();
        system.refresh_memory();
        Ok(shorthand::resolve_model_shorthand(&models, &identifier, system.total_memory())?.cloned())
    }

    #[bindings::export(Method)]
    pub async fn model_by_identifier(
        &self,
        identifier: ModelIdentifier,
    ) -> Result<Option<Model>, EngineError> {
        self.find_catalog_model(&identifier, |model| model.identifier == identifier).await
    }

    #[bindings::export(Method)]
    pub async fn model_by_repo_id(
        &self,
        repo_id: String,
    ) -> Result<Option<Model>, EngineError> {
        self.find_catalog_model(&repo_id, |model| model.repo_ids().contains(&repo_id)).await
    }

    #[bindings::export(Method)]
    pub async fn model_by_path(
        &self,
        path: String,
    ) -> Result<Option<Model>, EngineError> {
        let models = self.models().await?;
        let mut matches = Vec::new();
        for model in models {
            let candidate =
                model.filesystem_path().map(std::path::PathBuf::from).or_else(|| self.storage.cache_model_path(&model));
            if candidate.as_deref() != Some(Path::new(&path)) {
                continue;
            }
            if self.model_path(&model).await.is_some_and(|model_path| model_path == path) {
                matches.push(model);
            }
        }
        if let Some(model) = unique_model(&path, matches.into_iter())? {
            return Ok(Some(model));
        }
        if Path::new(&path).is_dir() {
            let backend = UzuLlmBackend::new();
            return LocalRegistry::model_at_path(Path::new(&path), backend.identifier(), backend.version())
                .map(Some)
                .map_err(EngineError::from);
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
        if !model.is_on_device() {
            return None;
        }
        if let Some(filesystem_path) = model.filesystem_path() {
            return Some(filesystem_path);
        }
        let state = self.storage.ready_state(&model.identifier).await.ok()?;
        if !matches!(state.phase, DownloadPhase::Downloaded {}) {
            return None;
        }
        self.storage.cache_model_path(model).map(|path| path.to_string_lossy().to_string())
    }

    #[bindings::export(Method)]
    pub fn downloader(
        &self,
        model: &Model,
    ) -> Downloader {
        Downloader::new(model.identifier.clone(), Arc::clone(&self.storage))
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
        downloader.resume().await?;
        downloader.progress().await
    }

    #[bindings::export(Method)]
    pub async fn download_state(
        &self,
        model: &Model,
    ) -> Option<DownloadState> {
        self.downloader(model).state().await
    }

    #[bindings::export(Method(Getter))]
    pub async fn download_states(&self) -> HashMap<ModelIdentifier, DownloadState> {
        self.storage.states().await
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
        let instance = self.chat_instance(model, config).await?;
        self.chat_with_instance(&instance).await
    }

    #[bindings::export(Method)]
    pub async fn chat_instance(
        &self,
        model: Model,
        config: ChatConfig,
    ) -> Result<ChatInstance, EngineError> {
        let path = self.model_path(&model).await;
        if let Some(backend) = model.backends.first() {
            let backend =
                self.backends.lock().await.get(&backend.identifier).ok_or(EngineError::BackendNotFound {})?.clone();
            let instance = ChatInstance::new(backend, config, model, path).await?;
            Ok(instance)
        } else {
            Err(EngineError::BackendNotFound {})
        }
    }

    #[bindings::export(Method)]
    pub async fn chat_with_instance(
        &self,
        instance: &ChatInstance,
    ) -> Result<ChatSession, EngineError> {
        let session = ChatSession::with_instance(instance).await?;
        Ok(session)
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
            let session = ClassificationSession::new(backend.clone(), model, path).await?;
            Ok(session)
        } else {
            Err(EngineError::BackendNotFound {})
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
            let session = TextToSpeechSession::new(backend.clone(), model, path).await?;
            Ok(session)
        } else {
            Err(EngineError::BackendNotFound {})
        }
    }
}

#[bindings::export(Implementation)]
impl Engine {
    #[bindings::export(Method)]
    pub async fn settings(&self) -> Result<Settings, EngineError> {
        self.settings.lock().await.clone().ok_or(EngineError::SettingsNotAvailable)
    }
}

impl Engine {
    pub fn storage_subscribe(&self) -> BroadcastStream<(ModelIdentifier, DownloadState)> {
        self.storage.subscribe()
    }

    pub async fn ready_download_state(
        &self,
        model: &Model,
    ) -> Result<DownloadState, EngineError> {
        Ok(self.storage.ready_state(&model.identifier).await?)
    }

    pub fn catalog_is_refreshing(&self) -> bool {
        self.catalog_refreshes.load(Ordering::SeqCst) != 0
    }

    pub async fn refresh_catalog(&self) -> Result<(), EngineError> {
        if !self.catalog_is_refreshing() {
            let registry = self.registry.lock().await.clone();
            let _ready = self.start_registry_refresh(Arc::new(registry)).await?;
        }
        Ok(())
    }

    pub fn catalog_subscribe(&self) -> BroadcastStream<()> {
        BroadcastStream::new(self.catalog_events.subscribe())
    }

    pub async fn catalog_snapshot(&self) -> Result<(Vec<Model>, bool), EngineError> {
        let (models, complete) = self.published_catalog.lock().await.clone();
        Ok((models, complete && self.catalog_refreshes.load(Ordering::SeqCst) == 0))
    }

    async fn handle_registry_refresh(
        &self,
        finished_refresh: bool,
    ) -> Result<(), EngineError> {
        let _publish = self.catalog_publish.lock().await;
        let registry = self.registry.lock().await.clone();
        let (models, complete) = registry.cached_listing().unwrap_or_default();
        let complete = complete && self.catalog_refreshes.load(Ordering::SeqCst) == usize::from(finished_refresh);
        // Register storage entries before exposing their descriptors. Individual
        // download verification continues asynchronously after registration.
        let result = self.storage.refresh(&models, complete).await;
        let mut published = self.published_catalog.lock().await;
        if result.is_ok() {
            *published = (models, complete);
        }
        // Completion and its snapshot become visible together: a lookup that
        // sees no active refresh must also read the final published snapshot.
        if finished_refresh {
            self.catalog_refreshes.fetch_sub(1, Ordering::SeqCst);
        }
        drop(published);
        let _ = self.catalog_events.send(());
        result?;
        if let Some(callback) = self.callback.lock().await.as_ref().cloned() {
            callback.on_event();
        };
        Ok(())
    }

    fn spawn_storage_listener(&self) {
        let mut stream = self.storage_subscribe();
        let callback = self.callback.clone();
        kiban::rt::spawn(async move {
            while let Some(update) = stream.next().await {
                let Ok(_) = update else {
                    continue;
                };
                if let Some(callback) = callback.lock().await.as_ref().cloned() {
                    callback.on_event();
                };
            }
        });
    }
}
