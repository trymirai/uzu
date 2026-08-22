mod config;
mod download_contents;
mod error;
mod hugging_face;
pub mod types;

use std::{
    collections::HashMap,
    fs::create_dir_all,
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

pub use config::Config;
pub use download_contents::DownloadContents;
use download_manager::{
    FileCheck, FileDownloadGroup, FileDownloadGroupSpec, FileDownloadManager, FileDownloadRequest, HttpDownloadRequest,
    RelativeFilePath, RequestHeaders,
};
pub use error::StorageError;
use futures_util::stream::select_all;
use hugging_face::{HuggingFaceDigest, HuggingFaceResolver, ResolvedHuggingFaceRepository};
use kiban::rt::{RuntimeHandle, TaskJoinHandle};
use shoji::types::{
    basic::File,
    model::{Model, ModelAccessibility, ModelReference},
};
use tokio::sync::{Mutex, broadcast::channel as tokio_broadcast_channel};
use tokio_stream::StreamExt;

use crate::{
    helpers::SharedAccess,
    storage::types::{DownloadState, Item, StorageDownloadEventSender, StorageDownloadEventStream},
};

pub struct Storage {
    pub config: Config,

    download_manager: Arc<dyn FileDownloadManager>,
    hugging_face: HuggingFaceResolver,
    catalog: SharedAccess<StorageCatalog>,
    refresh_lock: Mutex<()>,
    items_broadcast_sender: StorageDownloadEventSender,
    catalog_listener_task: SharedAccess<Option<Box<dyn TaskJoinHandle<()>>>>,
    handle: RuntimeHandle,
}

struct ResolvedModelDownload {
    cache_path: PathBuf,
    group_spec: FileDownloadGroupSpec,
}

#[derive(Clone, Default)]
struct StorageCatalog {
    items: HashMap<String, Item>,
    resolution_errors: HashMap<String, String>,
}

impl Storage {
    pub async fn new(
        runtime_handle: RuntimeHandle,
        config: Config,
    ) -> Result<Self, StorageError> {
        let cache_path = config.cache_path();
        let cache_path_error = || StorageError::UnableToCreateDirectory {
            path: cache_path.to_string_lossy().into_owned(),
        };
        reject_symlink_ancestors(&cache_path).map_err(|_| cache_path_error())?;
        create_dir_all(&cache_path).map_err(|_| cache_path_error())?;
        reject_symlink_ancestors(&cache_path).map_err(|_| cache_path_error())?;

        let download_manager = Arc::from(
            <dyn FileDownloadManager>::new(config.download_manager_type, runtime_handle.clone()).await.map_err(
                |error| StorageError::UnableToCreateDownloadManager {
                    message: error.to_string(),
                },
            )?,
        );
        let hugging_face =
            HuggingFaceResolver::new(config.huggingface_api_key().map(str::to_owned)).map_err(storage_error)?;
        let (items_broadcast_sender, _) = tokio_broadcast_channel(256);

        Ok(Self {
            config,
            download_manager,
            hugging_face,
            catalog: SharedAccess::new(StorageCatalog::default()),
            refresh_lock: Mutex::new(()),
            items_broadcast_sender,
            catalog_listener_task: SharedAccess::new(None),
            handle: runtime_handle,
        })
    }

    pub async fn refresh(
        &self,
        models: Vec<Model>,
    ) -> Result<(), StorageError> {
        let _refresh_guard = self.refresh_lock.lock().await;
        let models = models.into_iter().filter(Model::is_downloadable).collect::<Vec<_>>();
        let existing_items = self.catalog.lock().await.items.clone();
        let mut next_items = HashMap::with_capacity(models.len());
        let mut resolution_errors = HashMap::new();

        for model in models {
            let identifier = model.identifier.clone();
            let resolved = match self.resolve_model_download(&model).await {
                Ok(resolved) => resolved,
                Err(error) => {
                    tracing::warn!(model_identifier = identifier, %error, "model download could not be resolved");
                    if let Some(existing) = existing_items.get(&identifier).cloned() {
                        next_items.insert(identifier, existing);
                    } else {
                        resolution_errors.insert(identifier, error.to_string());
                    }
                    continue;
                },
            };

            if let Some(existing) = existing_items.get(&identifier)
                && existing.has_same_group_spec(&resolved.group_spec)
            {
                next_items.insert(identifier, existing.clone());
                continue;
            }

            let group = match FileDownloadGroup::open(Arc::clone(&self.download_manager), resolved.group_spec).await {
                Ok(group) => group,
                Err(error) => {
                    tracing::warn!(model_identifier = identifier, %error, "model download group could not be opened");
                    if let Some(existing) = existing_items.get(&identifier).cloned() {
                        next_items.insert(identifier, existing);
                    } else {
                        resolution_errors.insert(identifier, error.to_string());
                    }
                    continue;
                },
            };

            let item = Item::new(identifier.clone(), resolved.cache_path, group);
            next_items.insert(identifier, item);
        }

        let watched_items = next_items.values().cloned().collect();
        let previous_catalog = {
            let mut catalog = self.catalog.lock().await;
            std::mem::replace(
                &mut *catalog,
                StorageCatalog {
                    items: next_items,
                    resolution_errors,
                },
            )
        };

        self.replace_catalog_listener(watched_items).await;

        drop(previous_catalog);
        drop(existing_items);
        Ok(())
    }

    async fn replace_catalog_listener(
        &self,
        items: Vec<Item>,
    ) {
        let mut listener_task = self.catalog_listener_task.lock().await;
        if let Some(previous_task) = listener_task.take() {
            previous_task.abort_and_join().await;
        }

        if items.is_empty() {
            return;
        }

        let streams = items.into_iter().map(|item| {
            let identifier = item.identifier.clone();
            item.watch_states().map(move |state| (identifier.clone(), state))
        });
        let mut states = select_all(streams);
        let sender = self.items_broadcast_sender.clone();
        *listener_task = Some(self.handle.spawn(async move {
            while let Some(event) = states.next().await {
                let _ = sender.send(event);
            }
        }));
    }

    pub fn subscribe(&self) -> StorageDownloadEventStream {
        StorageDownloadEventStream::new(self.items_broadcast_sender.subscribe())
    }

    pub async fn get(
        &self,
        model_identifier: &str,
    ) -> Option<Item> {
        self.catalog.lock().await.items.get(model_identifier).cloned()
    }

    pub async fn state(
        &self,
        model_identifier: &str,
    ) -> Option<DownloadState> {
        let (item, error) = {
            let catalog = self.catalog.lock().await;
            (catalog.items.get(model_identifier).cloned(), catalog.resolution_errors.get(model_identifier).cloned())
        };
        match (item, error) {
            (Some(item), _) => Some(item.state().await),
            (None, Some(message)) => Some(DownloadState::error(message)),
            (None, None) => None,
        }
    }

    pub async fn states(&self) -> HashMap<String, DownloadState> {
        let catalog = self.catalog.lock().await.clone();
        let mut states = catalog
            .resolution_errors
            .into_iter()
            .map(|(identifier, message)| (identifier, DownloadState::error(message)))
            .collect::<HashMap<_, _>>();
        states.reserve(catalog.items.len());
        for (identifier, item) in catalog.items {
            states.insert(identifier, item.state().await);
        }
        states
    }

    pub async fn download(
        &self,
        model_identifier: &str,
    ) -> Result<(), StorageError> {
        self.item(model_identifier).await?.download().await
    }

    pub async fn pause(
        &self,
        model_identifier: &str,
    ) -> Result<(), StorageError> {
        self.item(model_identifier).await?.pause().await
    }

    pub async fn delete(
        &self,
        model_identifier: &str,
    ) -> Result<(), StorageError> {
        self.item(model_identifier).await?.cancel().await
    }

    async fn item(
        &self,
        identifier: &str,
    ) -> Result<Item, StorageError> {
        let catalog = self.catalog.lock().await;
        if let Some(item) = catalog.items.get(identifier) {
            return Ok(item.clone());
        }
        if let Some(message) = catalog.resolution_errors.get(identifier) {
            return Err(StorageError::ModelUnavailable {
                identifier: identifier.to_owned(),
                message: message.clone(),
            });
        }
        Err(StorageError::ItemNotFound {
            identifier: identifier.to_owned(),
        })
    }

    async fn resolve_model_download(
        &self,
        model: &Model,
    ) -> Result<ResolvedModelDownload, StorageError> {
        let ModelAccessibility::Local {
            reference,
            ..
        } = &model.accessibility
        else {
            return Err(StorageError::UnsupportedItem {
                identifier: model.identifier.clone(),
            });
        };

        match reference {
            ModelReference::Mirai {
                files,
                ..
            } => build_mirai_download(&self.config, model, files),
            ModelReference::HuggingFace {
                repository,
            } => {
                let resolved = self.hugging_face.resolve_repository(repository).await.map_err(storage_error)?;
                build_hugging_face_download(&self.config, model, resolved)
            },
            ModelReference::Local {
                ..
            } => Err(StorageError::UnsupportedItem {
                identifier: model.identifier.clone(),
            }),
        }
    }
}

fn build_mirai_download(
    config: &Config,
    model: &Model,
    all_files: &[File],
) -> Result<ResolvedModelDownload, StorageError> {
    let files = all_files.iter().filter(|file| config.download_contents.includes_file(&file.name));
    let mut requests = Vec::with_capacity(all_files.len());
    for file in files {
        let expected_bytes = u64::try_from(file.size).map_err(|_| StorageError::DownloadManager {
            message: format!("negative file size for {}", file.name),
        })?;
        let file_check = FileCheck::CRC(file.crc32c().ok_or_else(|| StorageError::HashNotFound {
            identifier: model.identifier.clone(),
            name: file.name.clone(),
        })?);
        requests.push(FileDownloadRequest::new(
            file.url.clone(),
            RelativeFilePath::try_from(file.name.as_str()).map_err(storage_error)?,
            file_check,
            Some(expected_bytes),
        ));
    }

    let revision = model.checkpoint_version().ok_or_else(|| StorageError::UnsupportedItem {
        identifier: model.identifier.clone(),
    })?;
    let cache_path = config.cache_model_path(model, &revision).ok_or_else(|| StorageError::UnsupportedItem {
        identifier: model.identifier.clone(),
    })?;
    let group_spec = FileDownloadGroupSpec::new(cache_path.clone(), requests).map_err(storage_error)?;
    ensure_binding_total_fits(&group_spec)?;
    Ok(ResolvedModelDownload {
        cache_path,
        group_spec,
    })
}

fn build_hugging_face_download(
    config: &Config,
    model: &Model,
    resolved: ResolvedHuggingFaceRepository,
) -> Result<ResolvedModelDownload, StorageError> {
    let mut requests = Vec::with_capacity(resolved.files.len());
    let headers = resolved.authorization.map(RequestHeaders::authorization).unwrap_or_default();

    for file in resolved.files {
        let relative_path = RelativeFilePath::try_from(file.relative_path.clone()).map_err(storage_error)?;
        let file_check = match file.digest {
            HuggingFaceDigest::Sha256(value) => FileCheck::Sha256(value),
            HuggingFaceDigest::GitBlobSha1(value) => FileCheck::GitBlobSha1(value),
        };
        requests.push(FileDownloadRequest::new(
            HttpDownloadRequest::with_headers(file.source_url, headers.clone()),
            relative_path,
            file_check,
            Some(file.size),
        ));
    }

    let cache_path = config.cache_model_path(model, &resolved.commit).ok_or_else(|| StorageError::UnsupportedItem {
        identifier: model.identifier.clone(),
    })?;
    let group_spec = FileDownloadGroupSpec::new(cache_path.clone(), requests).map_err(storage_error)?;
    ensure_binding_total_fits(&group_spec)?;
    Ok(ResolvedModelDownload {
        cache_path,
        group_spec,
    })
}

fn ensure_binding_total_fits(spec: &FileDownloadGroupSpec) -> Result<(), StorageError> {
    let total =
        spec.files().iter().filter_map(|file| file.expected_bytes).try_fold(0_u64, u64::checked_add).ok_or_else(
            || StorageError::DownloadManager {
                message: "model byte total overflow".to_string(),
            },
        )?;
    i64::try_from(total).map(|_| ()).map_err(|_| StorageError::DownloadManager {
        message: "model byte total exceeds the binding range".to_string(),
    })
}

fn reject_symlink_ancestors(path: &Path) -> io::Result<()> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() && !is_platform_path_alias(&current) => {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!("cache destination contains a symlink: {}", current.display()),
                ));
            },
            Ok(_) => {},
            Err(error) if error.kind() == io::ErrorKind::NotFound => break,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn is_platform_path_alias(path: &Path) -> bool {
    #[cfg(target_os = "macos")]
    {
        matches!(path.to_str(), Some("/var" | "/tmp" | "/etc"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = path;
        false
    }
}

fn storage_error(error: impl std::fmt::Display) -> StorageError {
    StorageError::DownloadManager {
        message: error.to_string(),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/storage/storage_test.rs"]
mod tests;
