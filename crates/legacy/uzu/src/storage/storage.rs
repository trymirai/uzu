use std::{collections::HashMap, path::PathBuf, sync::Arc};

use download_manager::{DownloadManager, DownloadState, DownloadTask, DownloadTaskRequest};
use futures_util::future::join_all;
use kiban::{fs, rt::RuntimeHandle};
use shoji::types::{
    basic::File,
    model::{Model, ModelAccessibility, ModelIdentifier, ModelReference},
};
use tokio::sync::{
    Mutex as TokioMutex,
    broadcast::{Sender as TokioBroadcastSender, channel as tokio_broadcast_channel},
};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

use crate::storage::{Config, StorageError, model_tasks::ModelTasks};

pub struct Storage {
    config: Config,
    download_manager: DownloadManager,
    tasks: TokioMutex<ModelTasks>,
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
            download_manager: DownloadManager::new(config.download_manager_type, runtime_handle),
            config,
            tasks: TokioMutex::new(ModelTasks::new()),
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
        let reference_name = model.reference_name()?;
        let checkpoint_version = model.checkpoint_version()?;
        Some(
            Self::cache_path(&self.config)
                .join("models")
                .join(reference_name)
                .join(model.cache_identifier())
                .join(checkpoint_version),
        )
    }

    pub async fn refresh(
        &self,
        models: &[Model],
    ) -> Result<(), StorageError> {
        let mut requests = HashMap::new();
        for model in models {
            let ModelAccessibility::Local {
                reference: ModelReference::Mirai {
                    files,
                    ..
                },
            } = &model.accessibility
            else {
                continue;
            };
            requests.entry(model.identifier.clone()).or_insert(self.request(model, files)?);
        }
        let missing: Vec<(ModelIdentifier, DownloadTaskRequest)> = {
            let mut tasks = self.tasks.lock().await;
            tasks.retain(|identifier, (task, forwarder)| {
                let keep = requests.get(identifier).is_some_and(|request| task.request() == request);
                if !keep {
                    forwarder.abort();
                }
                keep
            });
            requests.into_iter().filter(|(identifier, _)| !tasks.contains_key(identifier)).collect()
        };
        let created = join_all(missing.into_iter().map(|(identifier, request)| async move {
            (identifier, self.download_manager.download_task(request).await)
        }))
        .await;
        let mut tasks = self.tasks.lock().await;
        for (identifier, task) in created {
            let task = task?;
            let events = self.events.clone();
            let mut progress = task.progress();
            let forwarder = kiban::rt::spawn({
                let identifier = identifier.clone();
                async move {
                    while let Some(state) = progress.next().await {
                        let _ = events.send((identifier.clone(), state));
                    }
                }
            });
            if let Some((_, replaced)) = tasks.insert(identifier, (task, forwarder)) {
                replaced.abort();
            }
        }
        Ok(())
    }

    pub fn subscribe(&self) -> BroadcastStream<(ModelIdentifier, DownloadState)> {
        BroadcastStream::new(self.events.subscribe())
    }

    pub async fn state(
        &self,
        identifier: &ModelIdentifier,
    ) -> Result<DownloadState, StorageError> {
        Ok(self.model(identifier).await?.state())
    }

    pub async fn states(&self) -> HashMap<ModelIdentifier, DownloadState> {
        self.tasks.lock().await.iter().map(|(identifier, (task, _))| (identifier.clone(), task.state())).collect()
    }

    pub async fn download(
        &self,
        identifier: &ModelIdentifier,
    ) -> Result<(), StorageError> {
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
        self.tasks.lock().await.get(identifier).map(|(task, _)| Arc::clone(task)).ok_or_else(|| {
            StorageError::ModelNotFound {
                identifier: identifier.clone(),
            }
        })
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
                let crc32c = file.crc32c().ok_or_else(|| StorageError::HashNotFound {
                    identifier: model.identifier.clone(),
                    name: file.name.clone(),
                })?;
                Ok(DownloadTaskRequest::file()
                    .destination(&file.name)
                    .source_url(&file.url)
                    .expected_crc32c(crc32c)
                    .maybe_expected_bytes(u64::try_from(file.size).ok())
                    .build())
            })
            .collect::<Result<Vec<_>, StorageError>>()?;
        Ok(DownloadTaskRequest::group().destination(cache_path).subrequests(subrequests).build())
    }
}
