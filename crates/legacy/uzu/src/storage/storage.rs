use std::{collections::HashMap, path::PathBuf, sync::Arc};

use download_manager::{DestinationLock, DownloadManager, DownloadState, DownloadTask, DownloadTaskRequest};
use kiban::{fs, rt::RuntimeHandle};
use shoji::types::{
    basic::File,
    model::{Model, ModelAccessibility, ModelIdentifier, ModelSource},
};
use tokio::sync::{
    Mutex as TokioMutex,
    broadcast::{Sender as TokioBroadcastSender, channel as tokio_broadcast_channel},
};
use tokio_stream::wrappers::BroadcastStream;

use crate::{
    helpers::same_origin,
    storage::{
        Config, StorageError,
        model_tasks::{ModelTask, ModelTasks},
    },
};

pub struct Storage {
    config: Config,
    download_manager: Arc<DownloadManager>,
    tasks: TokioMutex<ModelTasks>,
    refresh_lock: TokioMutex<()>,
    events: TokioBroadcastSender<(ModelIdentifier, DownloadState)>,
}

impl Storage {
    pub async fn new(
        runtime_handle: RuntimeHandle,
        config: Config,
    ) -> Result<Self, StorageError> {
        let cache_path = Self::cache_path(&config);
        fs::asyn::create_dir_all(&cache_path).await.map_err(|_| StorageError::UnableToCreateDirectory {
            path: cache_path.to_string_lossy().to_string(),
        })?;
        let (events, _) = tokio_broadcast_channel(256);
        Ok(Self {
            download_manager: Arc::new(DownloadManager::new(config.download_manager_type, runtime_handle)),
            config,
            tasks: TokioMutex::new(ModelTasks::new()),
            refresh_lock: TokioMutex::new(()),
            events,
        })
    }

    pub fn cache_path(config: &Config) -> PathBuf {
        let home_path = PathBuf::from(config.device.home_path.clone());
        config.base_path.clone().unwrap_or(home_path).join(".cache").join(&config.name)
    }

    pub fn cache_model_path(
        &self,
        model: &Model,
    ) -> Option<PathBuf> {
        let checkpoint_version = model.checkpoint_version()?;
        Some(self.models_path().join(model.cache_identifier()).join(checkpoint_version))
    }

    pub async fn refresh(
        &self,
        models: &[Model],
        complete: bool,
    ) -> Result<(), StorageError> {
        let _refresh = self.refresh_lock.lock().await;
        let mut requests = HashMap::new();
        for model in models {
            let ModelAccessibility::OnDevice {
                source: ModelSource::Registry {
                    files,
                    ..
                },
            } = &model.accessibility
            else {
                continue;
            };
            let total_bytes = files.iter().map(|file| file.size.max(0)).fold(0_i64, i64::saturating_add);
            requests.entry(model.identifier.clone()).or_insert_with(|| (self.request(model, files), total_bytes));
        }
        let mut tasks = self.tasks.lock().await;
        let mut keep_paths: Vec<PathBuf> = requests
            .values()
            .filter_map(|(request, _)| request.as_ref().ok().map(|request| request.destination.clone()))
            .collect();
        let can_clean = complete && !keep_paths.is_empty();
        keep_paths.extend(
            tasks
                .values()
                .filter(|task| task.state().phase.is_in_progress())
                .filter_map(|task| task.request.as_ref().ok().map(|request| request.destination.clone())),
        );
        if complete {
            tasks.retain(|identifier, _| requests.contains_key(identifier));
        }
        for (identifier, (request, total_bytes)) in requests {
            if tasks.get(&identifier).is_some_and(|task| task.request == request) {
                continue;
            }
            let previous = tasks.remove(&identifier);
            tasks.insert(
                identifier.clone(),
                Arc::new(ModelTask::new(
                    identifier,
                    request,
                    total_bytes,
                    Arc::clone(&self.download_manager),
                    self.events.clone(),
                    previous,
                )),
            );
        }
        drop(tasks);
        if can_clean {
            self.remove_obsolete_checkpoints(&keep_paths).await;
        }
        Ok(())
    }

    async fn remove_obsolete_checkpoints(
        &self,
        keep_paths: &[PathBuf],
    ) {
        for model_path in fs::asyn::read_dir(self.models_path()).await.unwrap_or_default() {
            if !tokio::fs::symlink_metadata(&model_path).await.is_ok_and(|metadata| metadata.is_dir()) {
                continue;
            }
            let candidates = if keep_paths.iter().any(|path| path.starts_with(&model_path)) {
                fs::asyn::read_dir(&model_path).await.unwrap_or_default()
            } else {
                vec![model_path]
            };
            for path in candidates {
                if keep_paths.iter().any(|keep| keep.starts_with(&path))
                    || !tokio::fs::symlink_metadata(&path).await.is_ok_and(|metadata| metadata.is_dir())
                    || DestinationLock::held_within(&path).await
                {
                    continue;
                }
                if let Err(error) = fs::asyn::remove_dir_all(&path).await {
                    tracing::warn!(?error, path = %path.display(), "failed to remove obsolete checkpoint");
                }
            }
        }
    }

    pub fn subscribe(&self) -> BroadcastStream<(ModelIdentifier, DownloadState)> {
        BroadcastStream::new(self.events.subscribe())
    }

    pub async fn state(
        &self,
        identifier: &ModelIdentifier,
    ) -> Result<DownloadState, StorageError> {
        Ok(self.entry(identifier).await?.state())
    }

    pub async fn ready_state(
        &self,
        identifier: &ModelIdentifier,
    ) -> Result<DownloadState, StorageError> {
        Ok(self.model(identifier).await?.state())
    }

    pub async fn states(&self) -> HashMap<ModelIdentifier, DownloadState> {
        self.tasks.lock().await.iter().map(|(identifier, task)| (identifier.clone(), task.state())).collect()
    }

    pub async fn download(
        &self,
        identifier: &ModelIdentifier,
    ) -> Result<(), StorageError> {
        {
            let mut tasks = self.tasks.lock().await;
            if let Some(task) = tasks.get(identifier)
                && task.failed()
            {
                let replacement = Arc::new(ModelTask::new(
                    identifier.clone(),
                    task.request.clone(),
                    task.state().total_bytes,
                    Arc::clone(&self.download_manager),
                    self.events.clone(),
                    Some(Arc::clone(task)),
                ));
                tasks.insert(identifier.clone(), replacement);
            }
        }
        Ok(self.model(identifier).await?.download().await?)
    }

    pub async fn pause(
        &self,
        identifier: &ModelIdentifier,
    ) -> Result<(), StorageError> {
        Ok(self.model(identifier).await?.pause().await?)
    }

    pub async fn delete(
        &self,
        identifier: &ModelIdentifier,
    ) -> Result<(), StorageError> {
        Ok(self.model(identifier).await?.delete().await?)
    }

    async fn model(
        &self,
        identifier: &ModelIdentifier,
    ) -> Result<Arc<DownloadTask>, StorageError> {
        self.entry(identifier).await?.ready().await
    }

    async fn entry(
        &self,
        identifier: &ModelIdentifier,
    ) -> Result<Arc<ModelTask>, StorageError> {
        self.tasks.lock().await.get(identifier).cloned().ok_or_else(|| StorageError::ModelNotFound {
            identifier: identifier.clone(),
        })
    }

    fn models_path(&self) -> PathBuf {
        Self::cache_path(&self.config).join("models").join("mirai")
    }

    fn request(
        &self,
        model: &Model,
        files: &[File],
    ) -> Result<DownloadTaskRequest, StorageError> {
        let cache_path = self.cache_model_path(model).ok_or_else(|| StorageError::UnsupportedModel {
            identifier: model.identifier.clone(),
        })?;
        let subrequests = files
            .iter()
            .map(|file| {
                // Other origins reject a foreign bearer token: the Mirai CDN answers 401 to one.
                let bearer_token = self
                    .config
                    .huggingface_api_key
                    .clone()
                    .filter(|_| same_origin(&file.url, &self.config.huggingface_url));
                DownloadTaskRequest::file()
                    .destination(&file.name)
                    .source_url(&file.url)
                    .maybe_bearer_token(bearer_token)
                    .maybe_expected_bytes(u64::try_from(file.size).ok())
                    .build()
            })
            .collect();
        Ok(DownloadTaskRequest::group().destination(cache_path).subrequests(subrequests).build())
    }
}
