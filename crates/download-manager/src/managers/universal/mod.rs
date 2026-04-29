use crate::{
    DownloadError, DownloadEvent, FileCheck, FileDownloadManager, Path, SharedDownloadEventSender, TokioHandle,
    V2DownloadManagerAdapter,
};

#[derive(Debug, Clone)]
pub struct AsyncFetcherConfig {
    pub connections_per_file: u16,
    pub retries: u16,
    pub progress_interval_ms: u64,
}

impl Default for AsyncFetcherConfig {
    fn default() -> Self {
        Self {
            connections_per_file: 4,
            retries: 3,
            progress_interval_ms: 500,
        }
    }
}

impl AsyncFetcherConfig {
    pub fn new(
        connections_per_file: u16,
        retries: u16,
        progress_interval_ms: u64,
    ) -> Self {
        Self {
            connections_per_file,
            retries,
            progress_interval_ms,
        }
    }

    pub fn with_connections_per_file(
        mut self,
        connections_per_file: u16,
    ) -> Self {
        self.connections_per_file = connections_per_file;
        self
    }

    pub fn with_retries(
        mut self,
        retries: u16,
    ) -> Self {
        self.retries = retries;
        self
    }

    pub fn with_progress_interval_ms(
        mut self,
        progress_interval_ms: u64,
    ) -> Self {
        self.progress_interval_ms = progress_interval_ms;
        self
    }
}

pub struct AsyncFetcherDownloadManager {
    inner: V2DownloadManagerAdapter,
    pub config: AsyncFetcherConfig,
}

impl std::fmt::Debug for AsyncFetcherDownloadManager {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter.debug_struct("AsyncFetcherDownloadManager").field("config", &self.config).finish()
    }
}

impl AsyncFetcherDownloadManager {
    pub async fn new(
        config: AsyncFetcherConfig,
        tokio_handle: TokioHandle,
    ) -> Result<Self, DownloadError> {
        let inner = download_manager_v2::create_download_manager(
            download_manager_v2::FileDownloadManagerType::Universal,
            tokio_handle,
        )
        .await
        .map_err(|error| DownloadError::IOError(error.to_string()))?;
        Ok(Self {
            inner: V2DownloadManagerAdapter::new(inner),
            config,
        })
    }
}

#[async_trait::async_trait]
impl FileDownloadManager for AsyncFetcherDownloadManager {
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
