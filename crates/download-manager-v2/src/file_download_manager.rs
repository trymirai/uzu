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

    #[allow(clippy::ptr_arg)]
    async fn file_download_task(
        &self,
        source_url: &String,
        destination_path: &Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FileDownloadManagerType {
    Universal,
    #[default]
    Apple,
}

pub async fn create_download_manager(
    file_download_manager_type: FileDownloadManagerType,
    _tokio_handle: TokioHandle,
) -> Result<Box<dyn FileDownloadManager>, DownloadError> {
    match file_download_manager_type {
        FileDownloadManagerType::Universal => Ok(Box::new(crate::backends::universal::UniversalDownloadManager::new(
            "download-manager-v2-universal".to_string(),
        ))),
        FileDownloadManagerType::Apple => {
            #[cfg(target_vendor = "apple")]
            {
                Ok(Box::new(crate::backends::apple::AppleDownloadManager::new("download-manager-v2-apple".to_string())))
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                Err(DownloadError::UnsupportedType)
            }
        },
    }
}
