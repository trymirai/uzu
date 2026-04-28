use std::{path::Path, sync::Arc};

use tokio::{runtime::Handle as TokioHandle, sync::broadcast::Sender as TokioBroadcastSender};
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{DownloadError, DownloadId, FileCheck, FileDownloadEvent, FileDownloadTask};

pub type DownloadEvent = (DownloadId, FileDownloadEvent);
pub type DownloadEventSender = TokioBroadcastSender<DownloadEvent>;
pub type SharedDownloadEventSender = Arc<DownloadEventSender>;

#[async_trait::async_trait]
pub trait FileDownloadManager: Send + Sync + 'static {
    fn manager_id(&self) -> &str;

    fn subscribe_to_all_downloads(&self) -> TokioBroadcastStream<DownloadEvent>;

    fn global_broadcast_sender(&self) -> SharedDownloadEventSender;

    async fn get_all_file_tasks(&self) -> Result<Vec<Arc<dyn FileDownloadTask>>, DownloadError>;

    async fn file_download_task(
        &self,
        source_url: &String,
        destination_path: &Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileDownloadManagerType {
    Universal,
    Apple,
}

impl Default for FileDownloadManagerType {
    fn default() -> Self {
        if cfg!(target_vendor = "apple") {
            FileDownloadManagerType::Apple
        } else {
            FileDownloadManagerType::Universal
        }
    }
}

#[allow(unreachable_code)]
pub async fn create_download_manager(
    r#type: FileDownloadManagerType,
    tokio_handle: TokioHandle,
) -> Result<Box<dyn FileDownloadManager>, DownloadError> {
    match r#type {
        FileDownloadManagerType::Universal => {
            use crate::managers::universal::{AsyncFetcherConfig, AsyncFetcherDownloadManager};
            let config = AsyncFetcherConfig::default()
                .with_connections_per_file(4)
                .with_retries(3)
                .with_progress_interval_ms(500);
            let manager = AsyncFetcherDownloadManager::new(config, tokio_handle).await?;
            return Ok(Box::new(manager) as Box<dyn FileDownloadManager>);
        },
        FileDownloadManagerType::Apple => {
            #[cfg(target_vendor = "apple")]
            {
                use crate::managers::apple::{SessionConfig, URLSessionDownloadManager, URLSessionDropPolicy};
                let manager = URLSessionDownloadManager::new(
                    SessionConfig::default(),
                    URLSessionDropPolicy::FinishTasksAndInvalidate,
                    tokio_handle,
                )
                .await?;
                return Ok(Box::new(manager) as Box<dyn FileDownloadManager>);
            }
            return Err(DownloadError::UnsupportedType);
        },
    };
}
