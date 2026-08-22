use std::{path::PathBuf, sync::Arc};

use download_manager::{DownloadAttempt, FileDownloadGroup, FileDownloadGroupPhase, FileDownloadGroupState};
use shoji::types::basic::File;
use tokio::sync::broadcast::channel as tokio_broadcast_channel;
use tokio_stream::{Stream, StreamExt, wrappers::BroadcastStream};

use crate::storage::{
    StorageError,
    types::{DownloadPhase, DownloadState},
};

/// Binding-compatible model metadata around one shared file-download group.
#[derive(Clone)]
pub struct Item {
    pub identifier: String,
    pub files: Arc<Vec<File>>,
    pub cache_path: PathBuf,

    group: FileDownloadGroup,
}

impl std::fmt::Debug for Item {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter
            .debug_struct("Item")
            .field("identifier", &self.identifier)
            .field("cache_path", &self.cache_path)
            .field("group", &self.group)
            .finish()
    }
}

impl Item {
    pub fn new(
        identifier: String,
        files: Arc<Vec<File>>,
        cache_path: PathBuf,
        group: FileDownloadGroup,
    ) -> Self {
        Self {
            identifier,
            files,
            cache_path,
            group,
        }
    }

    pub async fn state(&self) -> DownloadState {
        binding_state(self.group.state())
    }

    /// Starts or retries this model download.
    ///
    /// The attempt handle is intentionally kept on the shared group API; this
    /// compatibility method preserves Item's previous `Result<()>` signature.
    pub async fn download(&self) -> Result<(), StorageError> {
        self.start_download().await.map(drop)
    }

    pub(crate) async fn start_download(&self) -> Result<DownloadAttempt, StorageError> {
        self.group.download().await.map_err(storage_error)
    }

    pub async fn pause(&self) -> Result<(), StorageError> {
        self.group.pause().await.map_err(storage_error)
    }

    pub async fn cancel(&self) -> Result<(), StorageError> {
        self.group.cancel().await.map_err(storage_error)
    }

    pub async fn progress(&self) -> Result<BroadcastStream<DownloadState>, StorageError> {
        let (sender, receiver) = tokio_broadcast_channel(64);
        let mut states = self.watch_states();
        kiban::rt::spawn(async move {
            while let Some(state) = states.next().await {
                if sender.send(state).is_err() {
                    break;
                }
            }
        });
        Ok(BroadcastStream::new(receiver))
    }

    pub(crate) fn watch_states(&self) -> impl Stream<Item = DownloadState> + Send + Unpin + 'static + use<> {
        self.group.subscribe().map(binding_state)
    }

    pub(crate) fn has_same_group_spec(
        &self,
        other: &download_manager::FileDownloadGroupSpec,
    ) -> bool {
        self.group.spec() == other
    }

    #[deprecated(note = "state() already returns the reduced group state")]
    pub async fn reduce_state(&self) -> DownloadState {
        self.state().await
    }

    #[deprecated(note = "group state is canonical and cannot be replaced externally")]
    pub async fn update_state_and_broadcast(
        &self,
        _new_state: DownloadState,
    ) {
    }

    #[deprecated(note = "per-file tasks are private implementation details of FileDownloadGroup")]
    #[allow(deprecated)]
    pub async fn file_task_by_download_id(
        &self,
        download_id: uuid::Uuid,
    ) -> Option<Arc<dyn download_manager::FileDownloadTask>> {
        self.group.legacy_file_task_by_download_id(download_id)
    }

    #[deprecated(note = "FileDownloadGroup::open reconciles state during construction")]
    pub async fn reconcile(&self) -> Result<(), StorageError> {
        let _ = self.state().await;
        Ok(())
    }

    #[deprecated(note = "pause() quiesces transfers while preserving resumable data")]
    pub async fn detach_active_downloads(&self) -> Result<(), StorageError> {
        if matches!(self.group.state().phase, FileDownloadGroupPhase::Downloading) {
            self.pause().await?;
        }
        Ok(())
    }

    #[deprecated(note = "Storage owns one merged group-state watcher")]
    pub async fn handle_file_task_update(&self) {}

    #[deprecated(note = "Storage owns one merged group-state watcher")]
    pub async fn start_listening(&self) {}

    #[deprecated(note = "Storage owns one merged group-state watcher")]
    pub async fn stop_listening(&self) {}
}

fn storage_error(error: impl std::fmt::Display) -> StorageError {
    StorageError::DownloadManager {
        message: error.to_string(),
    }
}

fn binding_state(state: FileDownloadGroupState) -> DownloadState {
    let total_bytes = state.total_bytes.and_then(|bytes| i64::try_from(bytes).ok()).unwrap_or(0);
    let Some(downloaded_bytes) = i64::try_from(state.downloaded_bytes).ok() else {
        return DownloadState::error("downloaded byte count exceeds the binding range".to_string());
    };

    match state.phase {
        FileDownloadGroupPhase::NotDownloaded => DownloadState::not_downloaded(total_bytes),
        FileDownloadGroupPhase::Downloading => DownloadState::downloading(downloaded_bytes, total_bytes),
        FileDownloadGroupPhase::Paused => DownloadState::paused(downloaded_bytes, total_bytes),
        FileDownloadGroupPhase::Downloaded => DownloadState::downloaded(total_bytes.max(downloaded_bytes)),
        FileDownloadGroupPhase::Locked => DownloadState::locked(downloaded_bytes, total_bytes),
        FileDownloadGroupPhase::Error => {
            let message = state
                .failures
                .iter()
                .map(|failure| format!("{}: {}", failure.relative_path, failure.error))
                .collect::<Vec<_>>()
                .join("; ");
            DownloadState {
                total_bytes,
                downloaded_bytes,
                phase: DownloadPhase::Error {
                    message,
                },
            }
        },
    }
}
