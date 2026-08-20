use std::{fmt::Debug, path::Path, sync::Arc};

use tokio::sync::broadcast::Sender as TokioBroadcastSender;
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{DownloadError, DownloadEventSender, DownloadId, FileCheck, FileDownloadState};

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
pub trait FileDownloadTask: Send + Sync + Debug {
    fn download_id(&self) -> DownloadId;
    fn source_url(&self) -> &str;
    fn destination(&self) -> &Path;
    fn file_check(&self) -> &FileCheck;
    fn expected_bytes(&self) -> Option<u64>;

    async fn download(&self) -> Result<(), DownloadError>;
    async fn pause(&self) -> Result<(), DownloadError>;
    async fn cancel(&self) -> Result<(), DownloadError>;
    async fn state(&self) -> FileDownloadState;
    async fn progress(&self) -> Result<TokioBroadcastStream<FileDownloadState>, DownloadError>;

    async fn start_listening(
        &self,
        global_broadcast: DownloadEventSender,
    );
    async fn stop_listening(&self);
    async fn wait(&self);

    fn broadcast_sender(&self) -> TokioBroadcastSender<FileDownloadState>;
}

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
pub(crate) trait ManagedFileDownloadTask: FileDownloadTask {
    async fn shutdown_for_removal(&self) -> Result<(), DownloadError>;
    fn is_stopped(&self) -> bool;
}

#[derive(Clone)]
pub(crate) struct CachedFileDownloadTask {
    public: Arc<dyn FileDownloadTask>,
    managed: Arc<dyn ManagedFileDownloadTask>,
}

impl CachedFileDownloadTask {
    pub(crate) fn new(
        public: Arc<dyn FileDownloadTask>,
        managed: Arc<dyn ManagedFileDownloadTask>,
    ) -> Self {
        Self {
            public,
            managed,
        }
    }

    pub(crate) fn public(&self) -> Arc<dyn FileDownloadTask> {
        Arc::clone(&self.public)
    }

    pub(crate) fn managed(&self) -> Arc<dyn ManagedFileDownloadTask> {
        Arc::clone(&self.managed)
    }

    pub(crate) fn is_stopped(&self) -> bool {
        self.managed.is_stopped()
    }
}
