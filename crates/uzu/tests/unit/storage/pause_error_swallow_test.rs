use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use download_manager::{
    DownloadError, DownloadEvent, DownloadEventSender, DownloadId, FileCheck, FileDownloadManager, FileDownloadState,
    FileDownloadTask, SharedDownloadEventSender,
};
use mock_registry::MockRegistry;
use tokio::{
    runtime::Handle as TokioHandle,
    sync::{
        Mutex as TokioMutex,
        broadcast::{Sender as TokioBroadcastSender, channel as tokio_broadcast_channel},
    },
};
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;
use uuid::Uuid;
use uzu::storage::types::{DownloadPhase, DownloadState, Item};

#[derive(Debug)]
struct PauseFailingFileTask {
    download_id: DownloadId,
    source_url: String,
    destination: PathBuf,
    state: TokioMutex<FileDownloadState>,
    broadcast_sender: TokioBroadcastSender<FileDownloadState>,
    file_check: FileCheck,
}

impl PauseFailingFileTask {
    fn new(
        source_url: String,
        destination: PathBuf,
        total_bytes: u64,
    ) -> Self {
        let (broadcast_sender, _) = tokio_broadcast_channel(64);
        Self {
            download_id: Uuid::new_v4(),
            source_url,
            destination,
            state: TokioMutex::new(FileDownloadState::downloading(total_bytes / 2, total_bytes)),
            broadcast_sender,
            file_check: FileCheck::None,
        }
    }
}

#[async_trait]
impl FileDownloadTask for PauseFailingFileTask {
    fn download_id(&self) -> DownloadId {
        self.download_id
    }
    fn source_url(&self) -> &str {
        &self.source_url
    }
    fn destination(&self) -> &std::path::Path {
        &self.destination
    }
    fn file_check(&self) -> &FileCheck {
        &self.file_check
    }
    async fn pause(&self) -> Result<(), DownloadError> {
        Err(DownloadError::Backend("forced pause failure".to_string()))
    }
    async fn state(&self) -> FileDownloadState {
        self.state.lock().await.clone()
    }
    async fn download(&self) -> Result<(), DownloadError> {
        unreachable!("Item::pause must not invoke file_task.download()");
    }
    async fn cancel(&self) -> Result<(), DownloadError> {
        unreachable!("Item::pause must not invoke file_task.cancel()");
    }
    async fn progress(&self) -> Result<TokioBroadcastStream<FileDownloadState>, DownloadError> {
        unreachable!("Item::pause must not invoke file_task.progress()");
    }
    async fn start_listening(
        &self,
        _: DownloadEventSender,
    ) {
        unreachable!("Item::pause must not invoke file_task.start_listening() when file_tasks are pre-populated");
    }
    async fn stop_listening(&self) {
        unreachable!("Item::pause must not invoke file_task.stop_listening()");
    }
    async fn wait(&self) {
        unreachable!("Item::pause must not invoke file_task.wait()");
    }
    fn broadcast_sender(&self) -> TokioBroadcastSender<FileDownloadState> {
        unreachable!("Item::pause must not invoke file_task.broadcast_sender()");
    }
}

#[derive(Debug)]
struct StubManager {
    sender: TokioBroadcastSender<DownloadEvent>,
}

impl StubManager {
    fn new() -> Self {
        let (sender, _) = tokio_broadcast_channel(64);
        Self {
            sender,
        }
    }
}

#[async_trait]
impl FileDownloadManager for StubManager {
    fn manager_id(&self) -> &str {
        "stub"
    }
    fn subscribe_to_all_downloads(&self) -> TokioBroadcastStream<DownloadEvent> {
        TokioBroadcastStream::new(self.sender.subscribe())
    }
    fn global_broadcast_sender(&self) -> SharedDownloadEventSender {
        Arc::new(self.sender.clone())
    }
    async fn get_all_file_tasks(&self) -> Result<Vec<Arc<dyn FileDownloadTask>>, DownloadError> {
        Ok(Vec::new())
    }
    async fn remove_file_task(
        &self,
        _: DownloadId,
    ) -> Result<(), DownloadError> {
        Ok(())
    }
    async fn file_download_task(
        &self,
        _: &String,
        _: &std::path::Path,
        _: FileCheck,
        _: Option<u64>,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError> {
        Err(DownloadError::Backend("not implemented in stub".to_string()))
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_pause_swallows_file_task_error_and_leaves_state_downloading()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;

    let temp_dir = tempfile::tempdir()?;
    let cache_path = temp_dir.path().join("model");

    let total_bytes = served_file.file.size as u64;
    let file_task = Arc::new(PauseFailingFileTask::new(
        served_file.file.url.clone(),
        cache_path.join(&served_file.file.name),
        total_bytes,
    ));

    let (storage_broadcast_sender, _storage_broadcast_receiver) = tokio_broadcast_channel(64);
    let item = Item::new(
        "test-model".to_string(),
        Arc::new(vec![served_file.file.clone()]),
        cache_path,
        DownloadState::downloading((total_bytes / 2) as i64, total_bytes as i64),
        Arc::new(StubManager::new()),
        vec![file_task as Arc<dyn FileDownloadTask>],
        TokioHandle::current(),
        storage_broadcast_sender,
    );

    let pause_result = item.pause().await;
    assert!(pause_result.is_err(), "Item::pause must propagate file_task pause errors instead of swallowing them");

    let state_after_pause = item.state().await;
    assert!(
        !matches!(state_after_pause.phase, DownloadPhase::Paused {}),
        "Item::pause failed, public phase must not falsely report Paused; got {:?}",
        state_after_pause.phase
    );

    Ok(())
}
