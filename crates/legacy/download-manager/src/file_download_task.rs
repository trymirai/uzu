use std::{
    fmt::Debug,
    path::{Path, PathBuf},
    sync::Arc,
};

use tokio::sync::{
    broadcast::{Sender as TokioBroadcastSender, channel as tokio_broadcast_channel, error::RecvError},
    watch::{Receiver as TokioWatchReceiver, channel as tokio_watch_channel},
};
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{
    DownloadError, DownloadEventSender, DownloadId, FileCheck, FileDownloadPhase, FileDownloadSnapshot,
    FileDownloadState, HttpDownloadRequest,
};

pub(crate) fn legacy_state_receiver(
    mut snapshots: TokioWatchReceiver<FileDownloadSnapshot>
) -> TokioWatchReceiver<FileDownloadState> {
    let initial_state = snapshots.borrow_and_update().state.clone();
    let (state_sender, state_receiver) = tokio_watch_channel(initial_state);
    kiban::rt::spawn(async move {
        while snapshots.changed().await.is_ok() {
            state_sender.send_replace(snapshots.borrow_and_update().state.clone());
        }
    });
    state_receiver
}

pub(crate) fn legacy_broadcast_sender(
    mut snapshots: TokioWatchReceiver<FileDownloadSnapshot>
) -> TokioBroadcastSender<FileDownloadState> {
    snapshots.borrow_and_update();
    let (state_sender, _) = tokio_broadcast_channel(64);
    let adapter_sender = state_sender.clone();
    kiban::rt::spawn(async move {
        while snapshots.changed().await.is_ok() {
            let _ = adapter_sender.send(snapshots.borrow_and_update().state.clone());
        }
    });
    state_sender
}

pub(crate) async fn wait_for_legacy_terminal(mut snapshots: TokioWatchReceiver<FileDownloadSnapshot>) {
    loop {
        let is_terminal = {
            let snapshot = snapshots.borrow_and_update();
            snapshot.failure.is_some()
                || matches!(
                    snapshot.state.phase,
                    FileDownloadPhase::Downloaded | FileDownloadPhase::Error(_) | FileDownloadPhase::LockedByOther(_)
                )
        };
        if is_terminal {
            return;
        }

        if snapshots.changed().await.is_err() {
            return;
        }
    }
}

#[allow(deprecated)]
pub(crate) async fn seeded_compatibility_snapshot_receiver(
    task: Arc<dyn FileDownloadTask>
) -> TokioWatchReceiver<FileDownloadSnapshot> {
    let mut source = task.snapshot_receiver();
    let state = task.state().await;
    let failure = task.failure();
    let synthesized =
        FileDownloadSnapshot::new(FileDownloadState::not_downloaded(task.expected_bytes().unwrap_or(0)), None);
    let source_snapshot = source.borrow_and_update().clone();
    let initial = if source_snapshot == synthesized {
        FileDownloadSnapshot::new(state, failure)
    } else {
        FileDownloadSnapshot::new(source_snapshot.state, source_snapshot.failure.or(failure))
    };
    let (snapshot_sender, snapshot_receiver) = tokio_watch_channel(initial);

    kiban::rt::spawn(async move {
        while source.changed().await.is_ok() {
            snapshot_sender.send_replace(source.borrow_and_update().clone());
        }
    });

    snapshot_receiver
}

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
pub trait FileDownloadTask: Send + Sync + Debug {
    fn download_id(&self) -> DownloadId;
    fn source_url(&self) -> &str;
    fn http_request(&self) -> HttpDownloadRequest {
        HttpDownloadRequest::get(self.source_url())
    }
    fn destination(&self) -> &Path;
    fn file_check(&self) -> &FileCheck;
    fn expected_bytes(&self) -> Option<u64>;

    async fn download(&self) -> Result<(), DownloadError>;
    async fn pause(&self) -> Result<(), DownloadError>;
    async fn cancel(&self) -> Result<(), DownloadError>;
    /// Destructive cleanup used by file groups.
    ///
    /// The default keeps implementations of the previous public trait source
    /// compatible. It first quiesces the task, then reports that the legacy
    /// implementation cannot prove ownership of every artifact.
    #[doc(hidden)]
    async fn cancel_and_delete(&self) -> Result<(), DownloadError> {
        self.cancel().await?;
        Err(DownloadError::DestructiveCleanupUnsupported)
    }
    async fn state(&self) -> FileDownloadState;

    /// Watches the complete state of this task.
    ///
    /// Implementations written before snapshots existed remain compatible: the
    /// default adapts their existing broadcast sender. New implementations
    /// should override this method so state and failure are published atomically.
    #[allow(deprecated)]
    fn snapshot_receiver(&self) -> TokioWatchReceiver<FileDownloadSnapshot> {
        let initial =
            FileDownloadSnapshot::new(FileDownloadState::not_downloaded(self.expected_bytes().unwrap_or(0)), None);
        let (snapshot_sender, snapshot_receiver) = tokio_watch_channel(initial);
        let mut legacy_events = self.broadcast_sender().subscribe();

        kiban::rt::spawn(async move {
            loop {
                match legacy_events.recv().await {
                    Ok(state) => {
                        snapshot_sender.send_replace(FileDownloadSnapshot::new(state, None));
                    },
                    Err(RecvError::Lagged(_)) => continue,
                    Err(RecvError::Closed) => break,
                }
            }
        });

        snapshot_receiver
    }

    /// Whether `snapshot_receiver()` is the task's canonical atomic watch.
    ///
    /// The default is false because implementations from before snapshot
    /// watches are adapted from broadcasts. File groups seed that compatibility
    /// stream from `state().await` before observing later updates.
    #[doc(hidden)]
    fn has_atomic_snapshot_watch(&self) -> bool {
        false
    }

    #[deprecated(note = "use snapshot_receiver() so state and typed failure are observed atomically")]
    fn state_receiver(&self) -> TokioWatchReceiver<FileDownloadState> {
        legacy_state_receiver(self.snapshot_receiver())
    }
    #[deprecated(note = "use snapshot_receiver() so state and failure are read atomically")]
    fn failure(&self) -> Option<DownloadError> {
        None
    }

    #[deprecated(note = "use snapshot_receiver()")]
    async fn progress(&self) -> Result<TokioBroadcastStream<FileDownloadState>, DownloadError>;

    #[deprecated(note = "subscribe to snapshot_receiver() and forward only at the compatibility boundary")]
    async fn start_listening(
        &self,
        global_broadcast: DownloadEventSender,
    );

    #[deprecated(note = "legacy listener compatibility API")]
    async fn stop_listening(&self);
    #[deprecated(note = "use the attempt-specific handle returned by FileDownloadGroup::download()")]
    async fn wait(&self);

    #[deprecated(note = "use snapshot_receiver()")]
    fn broadcast_sender(&self) -> TokioBroadcastSender<FileDownloadState>;
}

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
pub(crate) trait ManagedFileDownloadTask: FileDownloadTask {
    async fn shutdown_for_removal(&self) -> Result<(), DownloadError>;
    async fn shutdown_for_replacement_if_inactive(&self) -> Result<InactiveTaskShutdown, DownloadError>;
    async fn shutdown_preserving_artifacts_if_inactive(&self) -> Result<InactiveTaskShutdown, DownloadError>;
    fn is_stopped(&self) -> bool;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InactiveTaskShutdown {
    Stopped,
    Active,
}

#[derive(Clone)]
pub(crate) struct CachedFileDownloadTask {
    public: Arc<dyn FileDownloadTask>,
    managed: Arc<dyn ManagedFileDownloadTask>,
    artifact_root: PathBuf,
}

impl CachedFileDownloadTask {
    pub(crate) fn new(
        public: Arc<dyn FileDownloadTask>,
        managed: Arc<dyn ManagedFileDownloadTask>,
        artifact_root: PathBuf,
    ) -> Self {
        Self {
            public,
            managed,
            artifact_root,
        }
    }

    pub(crate) fn public(&self) -> Arc<dyn FileDownloadTask> {
        Arc::clone(&self.public)
    }

    pub(crate) fn managed(&self) -> Arc<dyn ManagedFileDownloadTask> {
        Arc::clone(&self.managed)
    }

    pub(crate) fn artifact_root(&self) -> &Path {
        &self.artifact_root
    }

    pub(crate) fn is_stopped(&self) -> bool {
        self.managed.is_stopped()
    }
}

#[cfg(test)]
#[path = "../tests/unit/file_download_task_test.rs"]
mod tests;
