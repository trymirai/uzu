use std::{
    collections::{HashMap, HashSet},
    fs::create_dir_all,
    path::PathBuf,
    sync::Arc,
};

use download_manager::{DownloadManager, DownloadState, DownloadTask, DownloadTaskRequest, FileCheck};
use futures_util::future::join_all;
use kiban::rt::{RuntimeHandle, TaskJoinHandle};
use shoji::types::{
    basic::File,
    model::{Model, ModelAccessibility, ModelIdentifier, ModelReference},
};
use tokio::sync::broadcast::{Sender as TokioBroadcastSender, channel as tokio_broadcast_channel};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

use crate::{
    helpers::SharedAccess,
    storage::{Config, StorageError, model_tasks::ModelTasks},
};

pub struct Storage {
    pub config: Config,

    download_manager: Box<dyn DownloadManager>,
    tasks: SharedAccess<ModelTasks>,
    events: TokioBroadcastSender<(ModelIdentifier, DownloadState)>,
}

impl Storage {
    pub async fn new(
        runtime_handle: RuntimeHandle,
        config: Config,
    ) -> Result<Self, StorageError> {
        let download_manager = <dyn DownloadManager>::new(config.download_manager_type, runtime_handle).await?;
        let (events, _) = tokio_broadcast_channel(256);
        let storage = Self {
            config,
            download_manager,
            tasks: SharedAccess::new(ModelTasks::new()),
            events,
        };
        let cache_path = storage.cache_path();
        create_dir_all(&cache_path).map_err(|_| StorageError::UnableToCreateDirectory {
            path: cache_path.to_string_lossy().to_string(),
        })?;
        Ok(storage)
    }

    pub fn cache_path(&self) -> PathBuf {
        let home_path = PathBuf::from(self.config.device.home_path.clone());
        self.config.base_path.clone().unwrap_or(home_path).join(".cache").join(&self.config.name)
    }

    pub fn cache_model_path(
        &self,
        model: &Model,
    ) -> Option<PathBuf> {
        let reference_name = model.reference_name()?;
        let checkpoint_version = model.checkpoint_version()?;
        Some(
            self.cache_path()
                .join("models")
                .join(reference_name)
                .join(model.cache_identifier())
                .join(checkpoint_version),
        )
    }

    pub fn log_name(&self) -> String {
        format!("{}.log", self.config.name)
    }

    pub async fn refresh(
        &self,
        models: Vec<Model>,
    ) -> Result<(), StorageError> {
        let models: Vec<Model> = models.into_iter().filter(Model::is_downloadable).collect();
        let identifiers: HashSet<&ModelIdentifier> = models.iter().map(|model| &model.identifier).collect();
        let missing: Vec<&Model> = {
            let mut tasks = self.tasks.lock().await;
            tasks.retain(|identifier, (_, forwarder)| {
                let keep = identifiers.contains(identifier);
                if !keep {
                    forwarder.abort();
                }
                keep
            });
            models.iter().filter(|model| !tasks.contains_key(&model.identifier)).collect()
        };
        let created = join_all(missing.iter().map(|model| self.task(model))).await;
        let mut tasks = self.tasks.lock().await;
        for (model, task) in missing.into_iter().zip(created) {
            let task = task?;
            let forwarder = self.forward(model.identifier.clone(), &task);
            tasks.insert(model.identifier.clone(), (task, forwarder));
        }
        Ok(())
    }

    pub fn subscribe(&self) -> BroadcastStream<(ModelIdentifier, DownloadState)> {
        BroadcastStream::new(self.events.subscribe())
    }

    pub async fn state(
        &self,
        identifier: &ModelIdentifier,
    ) -> Option<DownloadState> {
        Some(self.model(identifier).await.ok()?.state())
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
            StorageError::ItemNotFound {
                identifier: identifier.clone(),
            }
        })
    }

    async fn task(
        &self,
        model: &Model,
    ) -> Result<Arc<DownloadTask>, StorageError> {
        Ok(self.download_manager.download_task(self.request(model)?).await?)
    }

    fn request(
        &self,
        model: &Model,
    ) -> Result<DownloadTaskRequest, StorageError> {
        let cache_path = self.cache_model_path(model).ok_or_else(|| StorageError::UnsupportedItem {
            identifier: model.identifier.clone(),
        })?;
        let subrequests = model_files(model)?
            .iter()
            .map(|file| file_request(&model.identifier, file))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DownloadTaskRequest::group().destination(cache_path).subrequests(subrequests).build())
    }

    fn forward(
        &self,
        identifier: ModelIdentifier,
        task: &DownloadTask,
    ) -> Box<dyn TaskJoinHandle<()>> {
        let events = self.events.clone();
        let mut progress = task.progress();
        kiban::rt::spawn(async move {
            while let Some(result) = progress.next().await {
                if let Ok(state) = result {
                    let _ = events.send((identifier.clone(), state));
                }
            }
        })
    }
}

fn model_files(model: &Model) -> Result<&[File], StorageError> {
    match &model.accessibility {
        ModelAccessibility::Local {
            reference: ModelReference::Mirai {
                files,
                ..
            },
            ..
        } => Ok(files),
        _ => Err(StorageError::UnsupportedItem {
            identifier: model.identifier.clone(),
        }),
    }
}

fn file_request(
    identifier: &ModelIdentifier,
    file: &File,
) -> Result<DownloadTaskRequest, StorageError> {
    let crc32c = file.crc32c().ok_or_else(|| StorageError::HashNotFound {
        identifier: identifier.clone(),
        name: file.name.clone(),
    })?;
    Ok(DownloadTaskRequest::file()
        .destination(&file.name)
        .source_url(&file.url)
        .file_check(FileCheck::CRC(crc32c))
        .maybe_expected_bytes(u64::try_from(file.size).ok())
        .build())
}
