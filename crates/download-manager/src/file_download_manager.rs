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

#[bindings::export(Enumeration)]
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FileDownloadManagerType {
    Universal,
    #[cfg(target_vendor = "apple")]
    Apple,
}

impl Default for FileDownloadManagerType {
    #[allow(unreachable_code)]
    fn default() -> Self {
        #[cfg(target_vendor = "apple")]
        return Self::Apple;
        Self::Universal
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

    async fn file_download_task(
        &self,
        source_url: &str,
        destination_path: &Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError>;

    async fn destination_foreign_lock(
        &self,
        _destination_path: &Path,
    ) -> Option<String> {
        None
    }
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
            #[cfg(target_vendor = "apple")]
            FileDownloadManagerType::Apple => {
                let manager: Box<dyn FileDownloadManager> =
                    Box::new(AppleDownloadManager::from_tokio_handle(tokio_handle)?);
                Ok(manager)
            },
        }
    }

    pub async fn system_default(tokio_handle: TokioHandle) -> Result<Box<dyn FileDownloadManager>, DownloadError> {
        Self::new(FileDownloadManagerType::default(), tokio_handle).await
    }
}
