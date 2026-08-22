use std::{
    fmt::Debug,
    path::{Path, PathBuf},
    sync::Arc,
};

use tokio::sync::watch::Receiver as TokioWatchReceiver;

use crate::{DownloadError, DownloadId, FileCheck, FileDownloadSnapshot, FileDownloadState, HttpDownloadRequest};

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
    async fn cancel_and_delete(&self) -> Result<(), DownloadError>;
    async fn state(&self) -> FileDownloadState;

    fn snapshot_receiver(&self) -> TokioWatchReceiver<FileDownloadSnapshot>;
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
