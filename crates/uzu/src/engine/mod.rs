mod callback;
pub mod config;
mod download_manager;
mod downloader;
mod error;

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use backend_remote::openai::Backend as OpenAIBackend;
use backend_uzu::inference::Backend as UzuBackend;
pub use callback::{EngineCallback, EngineCallbackType};
pub use config::EngineConfig;
pub use download_manager::DownloadManagerType;
pub use downloader::{Downloader, DownloaderStream, DownloaderStreamUpdate};
pub use error::EngineError;
use indexmap::{IndexMap, IndexSet};
use nagare::{
    api::Config as ClientConfig,
    chat::ChatSession,
    classification::ClassificationSession,
    telemetry::{Telemetry, TelemetryContext, TelemetryDevice, TelemetryEvent},
    text_to_speech::TextToSpeechSession,
};
use shoji::{
    traits::{Backend, Registry},
    types::{
        model::{Model, ModelFamily, ModelRegistry, ModelVendor},
        session::chat::ChatConfig,
    },
};
use tokio::runtime::Handle;
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

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
    settings::Settings,
    storage::{
        Config as StorageConfig, Storage,
        types::{DownloadPhase, DownloadState},
    },
};

#[bindings::export(Class)]
#[derive(Clone)]
pub struct Engine {
    settings: SharedAccess<Option<Settings>>,
    registry: SharedAccess<MergedRegistry>,
    storage: SharedAccess<Storage>,
    backends: SharedAccess<HashMap<String, Arc<dyn Backend>>>,
    callback: SharedAccess<Option<Arc<EngineCallback>>>,
    telemetry: SharedAccess<Telemetry>,
    /// Whether anonymous usage telemetry may be reported. Defaults to `true`;
    /// consumers (e.g. an app with a privacy toggle) flip it via
    /// `set_usage_reporting`.
    report_usage: Arc<AtomicBool>,
}

impl Engine {
    pub async fn new(config: EngineConfig) -> Result<Self, EngineError> {
        let tokio_handle = Handle::try_current().map_err(|error| EngineError::TokioError {
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

        let telemetry = SharedAccess::new({
            let client_config = ClientConfig::new(
                "https://sdk.trymirai.com/api/v2".to_string(),
                Duration::from_secs(10),
                IndexMap::new(),
            );
            let context = TelemetryContext::new(
                env!("CARGO_PKG_VERSION").to_string(),
                backend_uzu::TOOLCHAIN_VERSION.to_string(),
                TelemetryDevice {
                    os_name: device.os_name.clone(),
                    cpu_name: device.cpu_name.clone(),
                    memory_total: device.memory_total,
                    is_environment_sandboxed: crate::device::is_environment_sandboxed(),
                },
            );
            Telemetry::new(client_config, "telemetry/events".to_string(), context)
        });

        let registry = SharedAccess::new(MergedRegistry::new(vec![]));
        let storage_config = StorageConfig::new(device.clone(), None, "mirai".to_string())
            .with_download_manager_type(config.download_manager_type.into());
        let storage_cache_path = storage_config.cache_path();
        logs::start(storage_config.cache_path(), &storage_config.log_name(), false);

        let storage = SharedAccess::new(Storage::new(tokio_handle, storage_config).await?);

        let engine = Self {
            settings: SharedAccess::new(settings),
            storage,
            registry,
            backends: SharedAccess::new(HashMap::new()),
            callback: SharedAccess::new(None),
            telemetry,
            report_usage: Arc::new(AtomicBool::new(true)),
        };
        engine.spawn_storage_listener().await;

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
                cache_path: storage_cache_path,
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
            engine.connect_openai(config).await?;
        }

        Ok(engine)
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
        self.registry.lock().await.add(Box::new(CachedRegistry::new(registry)))?;
        self.handle_registry_refresh().await?;
        Ok(())
    }

    pub async fn add_backend(
        &self,
        backend: Arc<dyn Backend>,
    ) {
        self.backends.lock().await.insert(backend.identifier(), backend);
    }

    /// Register (or replace) an OpenAI-compatible provider's registry *and*
    /// execution backend in one step, so a model connected at runtime is
    /// immediately usable — `add_registry` alone leaves `chat` failing with
    /// `BackendNotFound` until restart. Mirrors the provider setup `Engine::new`
    /// does at startup. The registry and backend are constructed *before* any
    /// existing provider is dropped, so an invalid config leaves it untouched.
    pub async fn connect_openai(
        &self,
        config: OpenAIConfig,
    ) -> Result<(), EngineError> {
        let registry = OpenAIRegistry::new(config.clone())?;
        let backend = OpenAIBackend::new(config.into()).map_err(|_| EngineError::UnableToCreateBackend {})?;
        let identifier = registry.indentifier();
        self.registry.lock().await.remove(&identifier)?;
        self.add_registry(Box::new(registry)).await?;
        self.add_backend(Arc::new(backend) as Arc<dyn Backend>).await;
        Ok(())
    }

    /// Enable or disable anonymous usage telemetry at runtime (e.g. from a
    /// privacy toggle). Affects download-event reporting; defaults to enabled.
    pub fn set_usage_reporting(
        &self,
        enabled: bool,
    ) {
        self.report_usage.store(enabled, Ordering::Relaxed);
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
        self.handle_registry_refresh().await?;
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
        self.registry.lock().await.models().await.map_err(EngineError::from)
    }

    #[bindings::export(Method(Getter))]
    pub async fn models_local(&self) -> Result<Vec<Model>, EngineError> {
        Ok(self.models().await?.into_iter().filter(|model| model.is_local()).collect())
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
        self.model_by_path(identifier).await
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
            if self.model_path(&model).await.is_some_and(|model_path| model_path == path) {
                return Ok(Some(model));
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
        if !model.is_local() {
            return None;
        }
        if let Some(local_external_path) = model.local_external_path() {
            return Some(local_external_path);
        }
        let storage = self.storage.lock().await;
        let state = storage.state(&model.identifier).await?;
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
    pub async fn download_states(&self) -> HashMap<String, DownloadState> {
        self.storage.lock().await.states().await
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
            let session =
                ChatSession::new(backend.clone(), config, model, path, self.telemetry.lock().await.clone()).await?;
            Ok(session)
        } else {
            Err(EngineError::BackendNotFound {})
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
    pub async fn storage_subscribe(&self) -> BroadcastStream<(String, DownloadState)> {
        self.storage.lock().await.subscribe()
    }

    async fn handle_registry_refresh(&self) -> Result<(), EngineError> {
        let models = self.registry.lock().await.models().await?;
        self.storage.lock().await.refresh(models).await?;
        if let Some(callback) = self.callback.lock().await.as_ref().cloned() {
            callback.on_event();
        };
        Ok(())
    }

    async fn spawn_storage_listener(&self) {
        let mut stream = self.storage_subscribe().await;
        let callback = self.callback.clone();
        let telemetry = self.telemetry.lock().await.clone();
        let report_usage = self.report_usage.clone();
        tokio::spawn(async move {
            let mut last_phase: HashMap<String, DownloadPhase> = HashMap::new();
            while let Some(update) = stream.next().await {
                let Ok((id, state)) = update else {
                    continue;
                };
                let previous = last_phase.insert(id.clone(), state.phase.clone());
                let event = match (&previous, &state.phase) {
                    (prev, DownloadPhase::Downloading {}) if !matches!(prev, Some(DownloadPhase::Downloading {})) => {
                        Some(TelemetryEvent::ModelDownloadStarted {
                            model_id: id,
                        })
                    },
                    (Some(DownloadPhase::Downloading {}), DownloadPhase::Downloaded {}) => {
                        Some(TelemetryEvent::ModelDownloadFinished {
                            model_id: id,
                        })
                    },
                    _ => None,
                };
                if let Some(event) = event
                    && report_usage.load(Ordering::Relaxed)
                {
                    telemetry.report(event);
                }
                if let Some(callback) = callback.lock().await.as_ref().cloned() {
                    callback.on_event();
                };
            }
        });
    }
}
