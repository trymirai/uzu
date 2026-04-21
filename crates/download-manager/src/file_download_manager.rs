use std::{path::Path, sync::Arc};

use tokio::sync::broadcast::Sender as TokioBroadcastSender;
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{DownloadError, DownloadId, FileCheck, FileDownloadEvent, FileDownloadTask};

#[async_trait::async_trait]
pub trait FileDownloadManager: Send + Sync + 'static {
    fn manager_id(&self) -> &str;

    fn subscribe_to_all_downloads(&self) -> TokioBroadcastStream<(DownloadId, FileDownloadEvent)>;

    fn global_broadcast_sender(&self) -> Arc<TokioBroadcastSender<(DownloadId, FileDownloadEvent)>>;

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
    identifier: Option<String>,
    tokio_handle: tokio::runtime::Handle,
) -> Result<Box<dyn FileDownloadManager>, DownloadError> {
    match r#type {
        FileDownloadManagerType::Universal => {
            use crate::managers::universal::{AsyncFetcherConfig, AsyncFetcherDownloadManager};
            let config = AsyncFetcherConfig::default()
                .with_connections_per_file(4)
                .with_retries(3)
                .with_progress_interval_ms(500);
            let manager = AsyncFetcherDownloadManager::new_with_manager_id(config, tokio_handle, identifier).await?;
            return Ok(Box::new(manager) as Box<dyn FileDownloadManager>);
        },
        FileDownloadManagerType::Apple => {
            #[cfg(target_vendor = "apple")]
            {
                use crate::managers::apple::{SessionConfig, URLSessionDownloadManager};
                let manager =
                    URLSessionDownloadManager::new_with_manager_id(SessionConfig::default(), tokio_handle, identifier)
                        .await?;
                return Ok(Box::new(manager) as Box<dyn FileDownloadManager>);
            }
            return Err(DownloadError::UnsupportedType);
        },
    };
}
