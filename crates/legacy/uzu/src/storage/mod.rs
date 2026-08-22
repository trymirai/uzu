mod config;
mod download_contents;
mod error;
mod hugging_face;
pub mod types;

use std::{
    collections::{HashMap, HashSet},
    fs::{File as FsFile, create_dir_all},
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
    files: Arc<Vec<File>>,
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
        let hugging_face = HuggingFaceResolver::new(
            cache_path.join("huggingface").join("trees"),
            config.huggingface_api_key().map(str::to_owned),
        )
        .map_err(storage_error)?;
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

            let item = Item::new(identifier.clone(), resolved.files, resolved.cache_path, group);
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
            } => self.resolve_mirai_download(model, files).await,
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

    async fn resolve_mirai_download(
        &self,
        model: &Model,
        all_files: &[File],
    ) -> Result<ResolvedModelDownload, StorageError> {
        let ResolvedModelDownload {
            files,
            cache_path,
            group_spec,
        } = build_mirai_download(&self.config, model, all_files)?;
        let cache_path = self.migrate_legacy_cache(model, cache_path, &files).await?;
        Ok(ResolvedModelDownload {
            files,
            cache_path,
            group_spec,
        })
    }

    async fn migrate_legacy_cache(
        &self,
        model: &Model,
        safe_path: PathBuf,
        files: &[File],
    ) -> Result<PathBuf, StorageError> {
        if std::fs::symlink_metadata(&safe_path).is_ok() {
            return Ok(safe_path);
        }
        let Some(legacy_path) = self.config.legacy_cache_model_path(model) else {
            return Ok(safe_path);
        };
        if std::fs::symlink_metadata(&legacy_path).is_err() {
            return Ok(safe_path);
        }

        let models_root = self.config.cache_models_path();
        reject_symlink_ancestors(&models_root).map_err(storage_error)?;
        create_dir_all(&models_root).map_err(storage_error)?;
        reject_symlink_ancestors(&models_root).map_err(storage_error)?;
        let Ok(relative_legacy) = legacy_path.strip_prefix(&models_root) else {
            return Ok(safe_path);
        };
        if !path_is_symlink_free(&models_root, relative_legacy) {
            return Ok(safe_path);
        }
        let Ok(canonical_root) = std::fs::canonicalize(&models_root) else {
            return Ok(safe_path);
        };
        let Ok(canonical_legacy) = std::fs::canonicalize(&legacy_path) else {
            return Ok(safe_path);
        };
        if !canonical_legacy.starts_with(&canonical_root) {
            return Ok(safe_path);
        }
        if !legacy_tree_is_safe(&canonical_legacy) {
            return Ok(safe_path);
        }
        let Some(verified_files) = verified_legacy_files(&canonical_legacy, files).await else {
            return Ok(safe_path);
        };
        if let Err(error) = install_verified_legacy_files(&canonical_legacy, &safe_path, &verified_files) {
            tracing::warn!(legacy_path = %canonical_legacy.display(), %error, "legacy model cache migration was skipped");
        }
        Ok(safe_path)
    }
}

fn build_mirai_download(
    config: &Config,
    model: &Model,
    all_files: &[File],
) -> Result<ResolvedModelDownload, StorageError> {
    let files =
        all_files.iter().filter(|file| config.download_contents.includes_file(&file.name)).cloned().collect::<Vec<_>>();
    let mut requests = Vec::with_capacity(files.len());
    for file in &files {
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

    let source_identity = canonical_source_identity(&requests)?;
    let revision = model.checkpoint_version().ok_or_else(|| StorageError::UnsupportedItem {
        identifier: model.identifier.clone(),
    })?;
    let cache_path = config.cache_model_path_for_source(model, &revision, &source_identity).ok_or_else(|| {
        StorageError::UnsupportedItem {
            identifier: model.identifier.clone(),
        }
    })?;
    let group_spec = FileDownloadGroupSpec::new(cache_path.clone(), requests).map_err(storage_error)?;
    ensure_binding_total_fits(&group_spec)?;
    Ok(ResolvedModelDownload {
        files: Arc::new(files),
        cache_path,
        group_spec,
    })
}

fn build_hugging_face_download(
    config: &Config,
    model: &Model,
    resolved: ResolvedHuggingFaceRepository,
) -> Result<ResolvedModelDownload, StorageError> {
    let mut files = Vec::with_capacity(resolved.files.len());
    let mut requests = Vec::with_capacity(resolved.files.len());
    let headers = resolved.authorization.map(RequestHeaders::authorization).unwrap_or_default();

    for file in resolved.files {
        let relative_path = RelativeFilePath::try_from(file.relative_path.clone()).map_err(storage_error)?;
        let file_check = match file.digest {
            HuggingFaceDigest::Sha256(value) => FileCheck::Sha256(value),
            HuggingFaceDigest::GitBlobSha1(value) => FileCheck::GitBlobSha1(value),
        };
        let size = i64::try_from(file.size).map_err(|_| StorageError::DownloadManager {
            message: format!("file size exceeds i64 for {relative_path}"),
        })?;
        files.push(File {
            url: file.source_url.clone(),
            name: relative_path.to_string(),
            size,
            hashes: Vec::new(),
        });
        requests.push(FileDownloadRequest::new(
            HttpDownloadRequest::with_headers(file.source_url, headers.clone()),
            relative_path,
            file_check,
            Some(file.size),
        ));
    }

    let source_identity = canonical_source_identity(&requests)?;
    let cache_path =
        config.cache_model_path_for_source(model, &resolved.commit, &source_identity).ok_or_else(|| {
            StorageError::UnsupportedItem {
                identifier: model.identifier.clone(),
            }
        })?;
    let group_spec = FileDownloadGroupSpec::new(cache_path.clone(), requests).map_err(storage_error)?;
    ensure_binding_total_fits(&group_spec)?;
    Ok(ResolvedModelDownload {
        files: Arc::new(files),
        cache_path,
        group_spec,
    })
}

async fn verified_legacy_files(
    root: &Path,
    files: &[File],
) -> Option<Vec<RelativeFilePath>> {
    if files.is_empty() {
        return None;
    }
    let root = std::fs::canonicalize(root).ok()?;
    let mut verified_files = Vec::with_capacity(files.len());
    let mut destinations = HashSet::with_capacity(files.len());
    for file in files {
        let relative_path = RelativeFilePath::try_from(file.name.as_str()).ok()?;
        if !destinations.insert(relative_path.as_path().to_path_buf()) {
            return None;
        };
        if !path_is_symlink_free(&root, relative_path.as_path()) {
            return None;
        }
        let destination = root.join(relative_path.as_path());
        let Ok(canonical_destination) = std::fs::canonicalize(&destination) else {
            return None;
        };
        if !canonical_destination.starts_with(&root) {
            return None;
        }
        let Ok(expected_bytes) = u64::try_from(file.size) else {
            return None;
        };
        let Ok(metadata) = tokio::fs::symlink_metadata(&canonical_destination).await else {
            return None;
        };
        if !metadata.is_file() || metadata.len() != expected_bytes {
            return None;
        }
        let crc = file.crc32c()?;
        if !download_manager::integrity_cache_matches(&canonical_destination, &FileCheck::CRC(crc)).await {
            return None;
        }
        verified_files.push(relative_path);
    }
    verified_files.sort_by(|left, right| left.as_path().cmp(right.as_path()));
    Some(verified_files)
}

fn path_is_symlink_free(
    root: &Path,
    relative_path: &Path,
) -> bool {
    let mut current = root.to_path_buf();
    let Ok(root_metadata) = std::fs::symlink_metadata(&current) else {
        return false;
    };
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return false;
    }
    for component in relative_path.components() {
        current.push(component);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => return false,
            Ok(_) => {},
            Err(_) => return false,
        }
    }
    true
}

fn legacy_tree_is_safe(root: &Path) -> bool {
    let Ok(root_metadata) = std::fs::symlink_metadata(root) else {
        return false;
    };
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return false;
    }
    let Ok(canonical_root) = std::fs::canonicalize(root) else {
        return false;
    };

    let mut pending = vec![canonical_root.clone()];
    let mut visited = HashSet::from([canonical_root.clone()]);
    while let Some(directory) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&directory) else {
            return false;
        };
        for entry in entries {
            let Ok(entry) = entry else {
                return false;
            };
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                return false;
            };
            if is_live_download_artifact(name) {
                return false;
            }

            let Ok(metadata) = std::fs::symlink_metadata(&path) else {
                return false;
            };
            if metadata.file_type().is_symlink() {
                return false;
            }
            if metadata.is_file() {
                if FsFile::open(&path).is_err() {
                    return false;
                }
                continue;
            }
            if !metadata.is_dir() {
                return false;
            }

            let Ok(canonical_directory) = std::fs::canonicalize(&path) else {
                return false;
            };
            if !canonical_directory.starts_with(&canonical_root) || !visited.insert(canonical_directory.clone()) {
                return false;
            }
            pending.push(canonical_directory);
        }
    }
    true
}

fn is_live_download_artifact(name: &str) -> bool {
    name.ends_with(".part")
        || name.ends_with(".resume_data")
        || name.ends_with(".lock")
        || name == "installing"
        || name.ends_with(".installing")
        || name.starts_with(".uzu-download-manager")
}

fn install_verified_legacy_files(
    legacy_root: &Path,
    safe_path: &Path,
    files: &[RelativeFilePath],
) -> io::Result<()> {
    let legacy_root = std::fs::canonicalize(legacy_root)?;
    let parent = safe_path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "safe cache path has no parent"))?;
    reject_symlink_ancestors(parent)?;
    create_dir_all(parent)?;
    reject_symlink_ancestors(parent)?;
    let safe_name = safe_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "safe cache path has no file name"))?;
    let staging_path = parent.join(format!(".{safe_name}.migrate-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&staging_path)?;
    reject_symlink_ancestors(&staging_path)?;

    let install_result = (|| {
        for relative_path in files {
            let source = legacy_root.join(relative_path.as_path());
            if !path_is_symlink_free(&legacy_root, relative_path.as_path()) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "legacy file path changed during migration"));
            }
            let canonical_source = std::fs::canonicalize(&source)?;
            if !canonical_source.starts_with(&legacy_root) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "legacy file escaped its cache root"));
            }

            let destination = staging_path.join(relative_path.as_path());
            if let Some(parent) = destination.parent() {
                reject_symlink_ancestors(parent)?;
                create_dir_all(parent)?;
                reject_symlink_ancestors(parent)?;
            }
            std::fs::copy(canonical_source, destination)?;
        }

        reject_symlink_ancestors(parent)?;
        reject_symlink_ancestors(&staging_path)?;
        match std::fs::rename(&staging_path, safe_path) {
            Ok(()) => Ok(()),
            Err(_)
                if std::fs::symlink_metadata(safe_path).is_ok_and(|metadata| metadata.is_dir())
                    && reject_symlink_ancestors(safe_path).is_ok() =>
            {
                Ok(())
            },
            Err(error) => Err(error),
        }
    })();

    if staging_path.exists() {
        let _ = std::fs::remove_dir_all(&staging_path);
    }
    install_result
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

#[derive(serde::Serialize)]
struct CanonicalDownloadMember<'a> {
    relative_path: String,
    source_url: &'a str,
    expected_bytes: Option<u64>,
    check: &'a FileCheck,
}

fn canonical_source_identity(requests: &[FileDownloadRequest]) -> Result<Vec<u8>, StorageError> {
    let mut members = requests
        .iter()
        .map(|request| CanonicalDownloadMember {
            relative_path: request.relative_path.to_string(),
            source_url: &request.source.url,
            expected_bytes: request.expected_bytes,
            check: &request.check,
        })
        .collect::<Vec<_>>();
    members.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    serde_json::to_vec(&members).map_err(storage_error)
}

fn storage_error(error: impl std::fmt::Display) -> StorageError {
    StorageError::DownloadManager {
        message: error.to_string(),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/storage/storage_test.rs"]
mod tests;
