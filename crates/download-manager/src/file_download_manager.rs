use std::{path::Path, sync::Arc};

use tokio::{runtime::Handle as TokioHandle, sync::broadcast::Sender as TokioBroadcastSender};
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{
    DownloadError, DownloadId, FileCheck, FileDownloadEvent, FileDownloadTask,
    backends::{apple::AppleDownloadManager, universal::UniversalDownloadManager},
};

pub type DownloadEvent = (DownloadId, FileDownloadEvent);
pub type DownloadEventSender = TokioBroadcastSender<DownloadEvent>;
pub type SharedDownloadEventSender = Arc<DownloadEventSender>;

#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FileDownloadManagerType {
    Universal,
    Apple,
}

impl Default for FileDownloadManagerType {
    fn default() -> Self {
        if cfg!(target_vendor = "apple") {
            Self::Apple
        } else {
            Self::Universal
        }
    }
}

#[async_trait::async_trait]
pub trait FileDownloadManager: Send + Sync + 'static {
    fn manager_id(&self) -> &str;
    fn subscribe_to_all_downloads(&self) -> TokioBroadcastStream<DownloadEvent>;
    fn global_broadcast_sender(&self) -> SharedDownloadEventSender;

    async fn get_all_file_tasks(&self) -> Result<Vec<Arc<dyn FileDownloadTask>>, DownloadError>;

    async fn remove_file_task(
        &self,
        download_id: DownloadId,
    ) -> Result<(), DownloadError>;

    #[allow(clippy::ptr_arg)]
    async fn file_download_task(
        &self,
        source_url: &String,
        destination_path: &Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError>;
}

impl dyn FileDownloadManager {
    pub async fn new(
        file_download_manager_type: FileDownloadManagerType,
        tokio_handle: TokioHandle,
    ) -> Result<Box<dyn FileDownloadManager>, DownloadError> {
        match file_download_manager_type {
            FileDownloadManagerType::Universal => {
                let manager: Box<dyn FileDownloadManager> =
                    Box::new(UniversalDownloadManager::from_tokio_handle(tokio_handle)?);
                Ok(manager)
            },
            FileDownloadManagerType::Apple => {
                #[cfg(target_vendor = "apple")]
                {
                    let manager: Box<dyn FileDownloadManager> =
                        Box::new(AppleDownloadManager::from_tokio_handle(tokio_handle)?);
                    Ok(manager)
                }
                #[cfg(not(target_vendor = "apple"))]
                {
                    Err(DownloadError::UnsupportedType)
                }
            },
        }
    }

    pub async fn system_default(tokio_handle: TokioHandle) -> Result<Box<dyn FileDownloadManager>, DownloadError> {
        Self::new(FileDownloadManagerType::default(), tokio_handle).await
    }
}
