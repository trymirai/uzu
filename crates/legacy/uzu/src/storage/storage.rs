use std::{
    collections::{HashMap, HashSet},
    fs::{create_dir_all, read_dir, remove_dir_all},
    sync::Arc,
};

use download_manager::{FileDownloadManager, FileDownloadPhase};
use futures_util::future::join_all;
use kiban::rt::RuntimeHandle;
use tokio::sync::broadcast::channel as tokio_broadcast_channel;

use super::{
    Config, StorageError,
    types::{DownloadPhase, DownloadState, Item, StorageDownloadEventSender, StorageDownloadEventStream},
};
use crate::{helpers::SharedAccess, models::ResolvedModels};

pub struct Storage {
    pub config: Config,

    download_manager: SharedAccess<Arc<dyn FileDownloadManager>>,
    items: SharedAccess<HashMap<String, Item>>,
    items_broadcast_sender: StorageDownloadEventSender,
    handle: RuntimeHandle,
}

impl Storage {
    pub async fn new(
        runtime_handle: RuntimeHandle,
        config: Config,
    ) -> Result<Self, StorageError> {
        create_dir_all(config.cache_path()).map_err(|_| StorageError::UnableToCreateDirectory {
            path: config.cache_path().to_string_lossy().to_string(),
        })?;

        let download_manager = SharedAccess::new(Arc::from(
            <dyn FileDownloadManager>::new(config.download_manager_type, runtime_handle.clone()).await.map_err(
                |error| StorageError::DownloadManager {
                    message: error.to_string(),
                },
            )?,
        ));

        let (items_broadcast_sender, _) = tokio_broadcast_channel(256);
        Ok(Self {
            config,
            download_manager,
            items: SharedAccess::new(HashMap::new()),
            items_broadcast_sender,
            handle: runtime_handle,
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/storage/mod.rs"]
mod tests;

impl Storage {
    pub async fn refresh(
        &self,
        models: &ResolvedModels,
        huggingface_api_key: Option<Arc<str>>,
    ) -> Result<(), StorageError> {
        let models = models
            .iter()
            .filter_map(|model| {
                let (model, files) = model.parts();
                files.map(|files| (model.clone(), Arc::new(files.to_vec())))
            })
            .collect::<Vec<_>>();
        let actual_model_identifiers: HashSet<String> =
            models.iter().map(|(model, _)| model.identifier.clone()).collect();

        let download_manager = { self.download_manager.lock().await.clone() };

        let existing_file_tasks =
            download_manager.get_all_file_tasks().await.map_err(|error| StorageError::DownloadManager {
                message: error.to_string(),
            })?;
        let mut active_file_tasks = Vec::with_capacity(existing_file_tasks.len());
        for task in existing_file_tasks {
            let task_state = task.state().await;
            if matches!(task_state.phase, FileDownloadPhase::Error(_)) {
                let download_id = task.download_id();
                let _ = task.cancel().await;
                let _ = download_manager.remove_file_task(download_id).await;
            } else {
                active_file_tasks.push(task);
            }
        }

        let stale_items = {
            let mut items = self.items.lock().await;
            items
                .extract_if(|identifier, _| !actual_model_identifiers.contains(identifier.as_str()))
                .map(|(_, item)| item)
                .collect::<Vec<_>>()
        };
        for item in stale_items {
            if let Err(error) = item.detach_active_downloads().await {
                tracing::warn!(?error, identifier = %item.identifier, "failed to detach stale model file tasks");
            }
            item.stop_listening().await;
        }

        for (model, files) in models {
            let identifier = model.identifier.clone();
            let cache_path = self.config.cache_model_path(&model).ok_or(StorageError::UnsupportedItem {
                identifier: identifier.clone(),
            })?;
            if let Some(item) = self.get(&identifier).await {
                if item.matches(&cache_path, &files) {
                    continue;
                }
                item.detach_active_downloads().await?;
                item.stop_listening().await;
            }

            let total_bytes = Item::total_bytes(&files)?;

            let has_files_on_disk = cache_path.exists();
            let has_active_tasks = files.iter().any(|file| {
                let file_path = cache_path.join(&file.file.name);
                active_file_tasks.iter().any(|task| task.destination() == file_path)
            });

            let item = Item::new(
                identifier.clone(),
                Arc::clone(&files),
                cache_path.clone(),
                DownloadState::not_downloaded(total_bytes),
                download_manager.clone(),
                self.handle.clone(),
                self.items_broadcast_sender.clone(),
            );

            if has_files_on_disk || has_active_tasks {
                item.ensure_file_tasks(huggingface_api_key.as_ref()).await?;
                item.start_listening().await;
                let _ = item.reconcile().await;
                let item_state = item.state().await;

                if matches!(item_state.phase, DownloadPhase::NotDownloaded {})
                    && cache_path.exists()
                    && let Ok(entries) = read_dir(&cache_path)
                {
                    let has_real_files = entries.flatten().any(|entry| {
                        entry
                            .file_name()
                            .to_str()
                            .is_some_and(|name| !name.ends_with(".resume_data") && !name.starts_with('.'))
                    });
                    if !has_real_files {
                        let _ = remove_dir_all(&cache_path);
                    }
                }
            }

            self.items.lock().await.insert(identifier, item);
        }

        Ok(())
    }

    pub fn subscribe(&self) -> StorageDownloadEventStream {
        StorageDownloadEventStream::new(self.items_broadcast_sender.subscribe())
    }

    pub async fn get(
        &self,
        model_identifier: &str,
    ) -> Option<Item> {
        let items = self.items.lock().await;
        items.get(model_identifier).cloned()
    }

    pub async fn state(
        &self,
        model_identifier: &str,
    ) -> Option<DownloadState> {
        let item = self.get(model_identifier).await?;
        let state = item.state().await;
        Some(state)
    }

    pub async fn states(&self) -> HashMap<String, DownloadState> {
        let items = self.items.lock().await;
        let state_futures = items.iter().map(|(identifier, item)| {
            let identifier = identifier.clone();
            async move { (identifier, item.state().await) }
        });
        join_all(state_futures).await.into_iter().collect()
    }

    pub async fn download(
        &self,
        model_identifier: &str,
        huggingface_api_key: Option<Arc<str>>,
    ) -> Result<(), StorageError> {
        let item = self.item(model_identifier).await?;
        match item.state().await.phase {
            DownloadPhase::Downloading {} | DownloadPhase::Downloaded {} => Ok(()),
            DownloadPhase::Error {
                ..
            } => {
                item.cancel().await?;
                item.download(huggingface_api_key).await
            },
            _ => item.download(huggingface_api_key).await,
        }
    }

    pub async fn pause(
        &self,
        model_identifier: &str,
    ) -> Result<(), StorageError> {
        let item = self.item(model_identifier).await?;
        item.pause().await
    }

    pub async fn delete(
        &self,
        model_identifier: &str,
    ) -> Result<(), StorageError> {
        let item = self.item(model_identifier).await?;
        item.cancel().await
    }

    async fn item(
        &self,
        model_identifier: &str,
    ) -> Result<Item, StorageError> {
        self.get(model_identifier).await.ok_or_else(|| StorageError::ItemNotFound {
            identifier: model_identifier.to_string(),
        })
    }
}
