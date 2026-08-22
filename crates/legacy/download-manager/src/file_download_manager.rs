use std::{path::Path, sync::Arc};

use kiban::rt::RuntimeHandle;

#[cfg(target_vendor = "apple")]
use crate::backends::apple::AppleDownloadManager;
use crate::{
    DownloadError, DownloadId, FileCheck, FileDownloadTask, HttpDownloadRequest,
    backends::universal::UniversalDownloadManager,
};

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

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
pub trait FileDownloadManager: Send + Sync + 'static {
    fn manager_id(&self) -> &str;
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
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError> {
        self.http_file_download_task(HttpDownloadRequest::get(source_url), destination_path, file_check, expected_bytes)
            .await
    }

    async fn http_file_download_task(
        &self,
        _request: HttpDownloadRequest,
        _destination_path: &Path,
        _file_check: FileCheck,
        _expected_bytes: Option<u64>,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError> {
        Err(DownloadError::Backend("HTTP request headers are unsupported by this download manager".to_string()))
    }

    #[doc(hidden)]
    async fn http_file_download_task_with_artifact_root(
        &self,
        request: HttpDownloadRequest,
        destination_path: &Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
        _artifact_root: &Path,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError> {
        self.http_file_download_task(request, destination_path, file_check, expected_bytes).await
    }

    /// Opens a child that already has recoverable local or backend state.
    ///
    /// Managers that can distinguish untouched downloads should return `None`
    /// so file groups can delay creating those children until `download()`.
    #[doc(hidden)]
    async fn open_existing_http_file_download_task_with_artifact_root(
        &self,
        request: HttpDownloadRequest,
        destination_path: &Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
        artifact_root: &Path,
    ) -> Result<Option<Arc<dyn FileDownloadTask>>, DownloadError> {
        self.http_file_download_task_with_artifact_root(
            request,
            destination_path,
            file_check,
            expected_bytes,
            artifact_root,
        )
        .await
        .map(Some)
    }

    /// Releases an exact cached child after its group retires, provided the
    /// child is no longer downloading.
    #[doc(hidden)]
    async fn release_file_task_if_inactive(
        &self,
        _task: Arc<dyn FileDownloadTask>,
    ) -> Result<(), DownloadError> {
        Ok(())
    }
}

impl dyn FileDownloadManager {
    pub async fn new(
        file_download_manager_type: FileDownloadManagerType,
        runtime_handle: RuntimeHandle,
    ) -> Result<Box<dyn FileDownloadManager>, DownloadError> {
        match file_download_manager_type {
            FileDownloadManagerType::Universal => {
                let manager: Box<dyn FileDownloadManager> =
                    Box::new(UniversalDownloadManager::from_runtime_handle(runtime_handle)?);
                Ok(manager)
            },
            #[cfg(target_vendor = "apple")]
            FileDownloadManagerType::Apple => {
                let manager: Box<dyn FileDownloadManager> =
                    Box::new(AppleDownloadManager::from_runtime_handle(runtime_handle)?);
                Ok(manager)
            },
        }
    }

    pub async fn system_default(runtime_handle: RuntimeHandle) -> Result<Box<dyn FileDownloadManager>, DownloadError> {
        Self::new(FileDownloadManagerType::default(), runtime_handle).await
    }
}
