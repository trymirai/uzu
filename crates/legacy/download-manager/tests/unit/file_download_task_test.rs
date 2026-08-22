use std::path::{Path, PathBuf};

use tokio_stream::StreamExt;

use super::*;

#[tokio::test]
async fn legacy_state_adapters_follow_atomic_snapshots() {
    let initial = FileDownloadSnapshot::new(FileDownloadState::not_downloaded(10), None);
    let (snapshot_sender, snapshot_receiver) = tokio_watch_channel(initial);
    let mut state_receiver = legacy_state_receiver(snapshot_receiver.clone());
    let broadcast_sender = legacy_broadcast_sender(snapshot_receiver);
    let mut progress = TokioBroadcastStream::new(broadcast_sender.subscribe());

    let downloading = FileDownloadState::downloading(4, 10);
    snapshot_sender.send_replace(FileDownloadSnapshot::new(downloading.clone(), None));

    state_receiver.changed().await.unwrap();
    assert_eq!(*state_receiver.borrow_and_update(), downloading);
    assert_eq!(progress.next().await.unwrap().unwrap(), downloading);
}

#[tokio::test]
async fn legacy_wait_finishes_from_a_terminal_snapshot() {
    let initial = FileDownloadSnapshot::new(FileDownloadState::not_downloaded(10), None);
    let (snapshot_sender, snapshot_receiver) = tokio_watch_channel(initial);
    let waiter = tokio::spawn(wait_for_legacy_terminal(snapshot_receiver));
    tokio::task::yield_now().await;
    assert!(!waiter.is_finished());

    let error = DownloadError::Backend("failed".to_string());
    snapshot_sender.send_replace(FileDownloadSnapshot::new(
        FileDownloadState::error_with_progress(0, 10, error.to_string()),
        Some(error),
    ));

    waiter.await.unwrap();
}

#[derive(Debug)]
struct PreviousVersionTask {
    destination: PathBuf,
    check: FileCheck,
    state: FileDownloadState,
    sender: TokioBroadcastSender<FileDownloadState>,
}

#[async_trait::async_trait]
impl FileDownloadTask for PreviousVersionTask {
    fn download_id(&self) -> DownloadId {
        DownloadId::nil()
    }

    fn source_url(&self) -> &str {
        "https://example.test/model.bin"
    }

    fn destination(&self) -> &Path {
        &self.destination
    }

    fn file_check(&self) -> &FileCheck {
        &self.check
    }

    fn expected_bytes(&self) -> Option<u64> {
        Some(10)
    }

    async fn download(&self) -> Result<(), DownloadError> {
        Ok(())
    }

    async fn pause(&self) -> Result<(), DownloadError> {
        Ok(())
    }

    async fn cancel(&self) -> Result<(), DownloadError> {
        Ok(())
    }

    async fn state(&self) -> FileDownloadState {
        self.state.clone()
    }

    async fn progress(&self) -> Result<TokioBroadcastStream<FileDownloadState>, DownloadError> {
        Ok(TokioBroadcastStream::new(self.sender.subscribe()))
    }

    async fn start_listening(
        &self,
        _global_broadcast: DownloadEventSender,
    ) {
    }

    async fn stop_listening(&self) {}

    async fn wait(&self) {}

    fn broadcast_sender(&self) -> TokioBroadcastSender<FileDownloadState> {
        self.sender.clone()
    }
}

#[tokio::test]
#[allow(deprecated)]
async fn previous_trait_implementations_use_compatibility_defaults() {
    let (sender, _) = tokio_broadcast_channel(8);
    let task = PreviousVersionTask {
        destination: PathBuf::from("model.bin"),
        check: FileCheck::None,
        state: FileDownloadState::not_downloaded(10),
        sender,
    };
    let mut snapshots = task.snapshot_receiver();
    let mut states = task.state_receiver();
    let downloading = FileDownloadState::downloading(4, 10);

    task.sender.send(downloading.clone()).unwrap();
    snapshots.changed().await.unwrap();
    states.changed().await.unwrap();

    assert_eq!(snapshots.borrow_and_update().state, downloading);
    assert_eq!(*states.borrow_and_update(), downloading);
    assert_eq!(task.cancel_and_delete().await, Err(DownloadError::DestructiveCleanupUnsupported));
}

#[tokio::test]
async fn compatibility_snapshot_is_seeded_from_an_existing_terminal_state() {
    let (sender, _) = tokio_broadcast_channel(8);
    let task: Arc<dyn FileDownloadTask> = Arc::new(PreviousVersionTask {
        destination: PathBuf::from("model.bin"),
        check: FileCheck::None,
        state: FileDownloadState::downloaded(10),
        sender,
    });

    let snapshots = seeded_compatibility_snapshot_receiver(task).await;

    assert_eq!(snapshots.borrow().state, FileDownloadState::downloaded(10));
}
