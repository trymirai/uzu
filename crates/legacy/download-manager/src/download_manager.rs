use std::sync::Arc;

use kiban::rt::RuntimeHandle;

#[cfg(target_vendor = "apple")]
use crate::backends::apple::AppleDownloadManager;
use crate::{
    DownloadError, DownloadManagerType, DownloadTask, DownloadTaskRequest,
    backends::universal::UniversalDownloadManager,
};

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
pub trait DownloadManager: Send + Sync + 'static {
    fn manager_id(&self) -> &str;

    async fn download_task(
        &self,
        request: DownloadTaskRequest,
    ) -> Result<Arc<DownloadTask>, DownloadError>;
}

impl dyn DownloadManager {
    pub async fn new(
        download_manager_type: DownloadManagerType,
        runtime_handle: RuntimeHandle,
    ) -> Result<Box<dyn DownloadManager>, DownloadError> {
        match download_manager_type {
            DownloadManagerType::Universal => {
                Ok(Box::new(UniversalDownloadManager::from_runtime_handle(runtime_handle)?))
            },
            #[cfg(target_vendor = "apple")]
            DownloadManagerType::Native => Ok(Box::new(AppleDownloadManager::from_runtime_handle(runtime_handle)?)),
        }
    }

    pub async fn system_default(runtime_handle: RuntimeHandle) -> Result<Box<dyn DownloadManager>, DownloadError> {
        Self::new(DownloadManagerType::default(), runtime_handle).await
    }
}
