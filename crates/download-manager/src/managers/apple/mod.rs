use crate::{
    DownloadError, DownloadEvent, FileCheck, FileDownloadManager, Path, SharedDownloadEventSender, TokioHandle,
    V2DownloadManagerAdapter,
};
use crate::prelude::NSURLSessionTaskState;

pub type URLSessionDownloadTaskState = Option<NSURLSessionTaskState>;

#[derive(Debug, Clone)]
pub enum BackgroundSessionID {
    Default,
    Custom(String),
}

#[derive(Debug, Clone, Default)]
pub enum SessionConfig {
    Foreground,
    Background(BackgroundSessionID),
    #[default]
    Automatic,
}

#[derive(Debug, Clone)]
pub enum URLSessionDropPolicy {
    FinishTasksAndInvalidate,
    InvalidateAndCancel,
}

pub struct URLSessionDownloadManager {
    inner: V2DownloadManagerAdapter,
    _session_config: SessionConfig,
    _drop_policy: URLSessionDropPolicy,
}

impl std::fmt::Debug for URLSessionDownloadManager {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter.debug_struct("URLSessionDownloadManager").finish_non_exhaustive()
    }
}

impl URLSessionDownloadManager {
    pub async fn new(
        session_config: SessionConfig,
        drop_policy: URLSessionDropPolicy,
        tokio_handle: TokioHandle,
    ) -> Result<Self, DownloadError> {
        let inner =
            download_manager_v2::create_download_manager(download_manager_v2::FileDownloadManagerType::Apple, tokio_handle)
                .await
                .map_err(|error| DownloadError::IOError(error.to_string()))?;
        Ok(Self {
            inner: V2DownloadManagerAdapter::new(inner),
            _session_config: session_config,
            _drop_policy: drop_policy,
        })
    }
}

#[async_trait::async_trait]
impl FileDownloadManager for URLSessionDownloadManager {
    fn manager_id(&self) -> &str {
        self.inner.manager_id()
    }

    fn subscribe_to_all_downloads(&self) -> crate::prelude::TokioBroadcastStream<DownloadEvent> {
        self.inner.subscribe_to_all_downloads()
    }

    fn global_broadcast_sender(&self) -> SharedDownloadEventSender {
        self.inner.global_broadcast_sender()
    }

    async fn get_all_file_tasks(&self) -> Result<Vec<crate::Arc<dyn crate::FileDownloadTask>>, DownloadError> {
        self.inner.get_all_file_tasks().await
    }

    #[allow(clippy::ptr_arg)]
    async fn file_download_task(
        &self,
        source_url: &String,
        destination_path: &Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
    ) -> Result<crate::Arc<dyn crate::FileDownloadTask>, DownloadError> {
        self.inner
            .file_download_task(source_url, destination_path, file_check, expected_bytes)
            .await
    }
}

pub type FileDownloadTask = dyn crate::FileDownloadTask;

#[derive(Debug, Clone)]
pub struct URLSessionDelegate;

#[derive(Debug, Clone)]
pub struct URLSessionDownloadTaskResumeData;

#[derive(Debug, Clone)]
pub struct URLSessionError;

#[derive(Debug)]
pub struct URLSessionGetTasksCompletionHandler;

#[derive(Debug)]
pub struct URLSessionResumeDataHandler;
