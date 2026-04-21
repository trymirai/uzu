mod config;
mod error;
pub mod types;

use std::{
    collections::{HashMap, HashSet},
    fs::{create_dir_all, read_dir, remove_dir_all},
    sync::Arc,
};

pub use config::Config;
use download_manager::{
    FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase, create_download_manager,
};
pub use error::Error;
use shoji::types::model::{Accessibility, File, Model, Reference};
use tokio::{
    runtime::Handle,
    sync::broadcast::{Sender, channel},
};
use tokio_stream::wrappers::BroadcastStream;

use crate::{
    helpers::SharedAccess,
    storage::types::{DownloadPhase, DownloadState, Item},
};

pub struct Storage {
    pub config: Config,

    download_manager: SharedAccess<Arc<dyn FileDownloadManager>>,
    items: SharedAccess<HashMap<String, Item>>,
    items_broadcast_sender: Sender<(String, DownloadState)>,
    handle: Handle,
}

impl Storage {
    pub async fn new(
        tokio_handle: Handle,
        config: Config,
    ) -> Result<Self, Error> {
        create_dir_all(config.cache_path()).map_err(|_| Error::UnableToCreateDirectory {
            path: config.cache_path().to_string_lossy().to_string(),
        })?;

        let download_manager = SharedAccess::new(Arc::from(
            create_download_manager(
                FileDownloadManagerType::default(),
                Some(config.name.clone()),
                tokio_handle.clone(),
            )
            .await
            .map_err(|error| Error::DownloadManager {
                message: error.to_string(),
            })?,
        ));

        let items = SharedAccess::new(HashMap::new());

        let (items_broadcast_sender, _) = channel(256);

        let storage = Self {
            config,
            download_manager,
            items,
            items_broadcast_sender,
            handle: tokio_handle,
        };
        Ok(storage)
    }
}

impl Storage {
    pub async fn refresh(
        &self,
        models: Vec<Model>,
    ) -> Result<(), Error> {
        let models = models.into_iter().filter(|model| model.is_downloadable()).collect::<Vec<_>>();
        let actual_model_identifiers: HashSet<String> = models.iter().map(|model| model.identifier()).collect();

        let download_manager = { self.download_manager.lock().await.clone() };

        let existing_file_tasks =
            download_manager.get_all_file_tasks().await.map_err(|error| Error::DownloadManager {
                message: error.to_string(),
            })?;
        let mut active_file_tasks = Vec::with_capacity(existing_file_tasks.len());
        for task in existing_file_tasks {
            let task_state = task.state().await;
            if matches!(task_state.phase, FileDownloadPhase::Error(_)) {
                let _ = task.cancel().await;
            } else {
                active_file_tasks.push(task);
            }
        }

        let mut items = self.items.lock().await;
        let stale_model_identifiers: Vec<String> = items
            .keys()
            .filter(|identifier| !actual_model_identifiers.contains(identifier.as_str()))
            .cloned()
            .collect();
        for identifier in stale_model_identifiers {
            if let Some(item) = items.remove(&identifier) {
                item.stop_listening().await;
            }
        }

        for model in models {
            let identifier = model.identifier();
            if items.contains_key(&identifier) {
                continue;
            }

            let files = self.resolve_model_files(&model)?;
            let total_bytes: u64 = files.iter().map(|file| file.size as u64).sum();

            let cache_path = self.config.cache_model_path(&model).ok_or(Error::UnsupportedItem {
                identifier: identifier.clone(),
            })?;

            let has_files_on_disk = cache_path.exists();
            let has_active_tasks = files.iter().any(|file| {
                let file_path = cache_path.join(&file.name);
                active_file_tasks.iter().any(|task| task.destination() == file_path)
            });

            let item = if has_files_on_disk || has_active_tasks {
                let mut file_tasks = Vec::new();
                for file in &files {
                    let file_path = cache_path.join(&file.name);
                    let file_check = FileCheck::CRC(file.crc32c().ok_or(Error::HashNotFound {
                        identifier: identifier.clone(),
                        name: file.name.clone(),
                    })?);

                    let file_task = download_manager
                        .file_download_task(&file.url, &file_path, file_check, Some(file.size as u64))
                        .await
                        .map_err(|error| Error::DownloadManager {
                            message: error.to_string(),
                        })?;

                    file_tasks.push(file_task);
                }

                let item = Item::new(
                    identifier.clone(),
                    files.into(),
                    cache_path.clone(),
                    DownloadState::not_downloaded(total_bytes as i64),
                    download_manager.clone(),
                    file_tasks,
                    self.handle.clone(),
                    self.items_broadcast_sender.clone(),
                );

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

                item
            } else {
                Item::new(
                    identifier.clone(),
                    files.into(),
                    cache_path.clone(),
                    DownloadState::not_downloaded(total_bytes as i64),
                    download_manager.clone(),
                    Vec::new(),
                    self.handle.clone(),
                    self.items_broadcast_sender.clone(),
                )
            };

            items.insert(identifier, item);
        }

        Ok(())
    }

    pub fn subscribe(&self) -> BroadcastStream<(String, DownloadState)> {
        BroadcastStream::new(self.items_broadcast_sender.subscribe())
    }

    pub async fn get(
        &self,
        model_identifier: &String,
    ) -> Option<Item> {
        let items = self.items.lock().await;
        items.get(model_identifier).cloned()
    }

    pub async fn state(
        &self,
        model_identifier: &String,
    ) -> Option<DownloadState> {
        let item = self.get(model_identifier).await?;
        let state = item.state().await;
        Some(state)
    }

    pub async fn download(
        &self,
        model_identifier: &String,
    ) -> Result<(), Error> {
        let item = self.get(model_identifier).await.ok_or_else(|| Error::ItemNotFound {
            identifier: model_identifier.clone(),
        })?;
        item.download().await
    }

    pub async fn pause(
        &self,
        model_identifier: &String,
    ) -> Result<(), Error> {
        let item = self.get(model_identifier).await.ok_or_else(|| Error::ItemNotFound {
            identifier: model_identifier.clone(),
        })?;
        item.pause().await
    }

    pub async fn delete(
        &self,
        model_identifier: &String,
    ) -> Result<(), Error> {
        let item = self.get(model_identifier).await.ok_or_else(|| Error::ItemNotFound {
            identifier: model_identifier.clone(),
        })?;
        item.cancel().await
    }
}

impl Storage {
    fn resolve_model_files(
        &self,
        model: &Model,
    ) -> Result<Vec<File>, Error> {
        match &model.accessibility {
            Accessibility::Local {
                reference,
                ..
            } => match &reference {
                Reference::Mirai {
                    files,
                    ..
                } => Ok(files.clone()),
                Reference::HuggingFace {
                    ..
                } => Err(Error::UnsupportedItem {
                    identifier: model.identifier(),
                }),
                Reference::Local {
                    ..
                } => Err(Error::UnsupportedItem {
                    identifier: model.identifier(),
                }),
            },
            Accessibility::Remote {
                ..
            } => Err(Error::UnsupportedItem {
                identifier: model.identifier(),
            }),
        }
    }
}
