use std::collections::HashMap;

use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{
    Arc, DownloadError, DownloadId, FileCheck, FileDownloadEvent, FileDownloadManager,
    FileDownloadTask as FileDownloadTaskTrait, Path, PathBuf, TokioBroadcastSender, TokioHandle, TokioMutex,
    compute_download_id,
    managers::universal::{AsyncFetcherConfig, FileDownloadTask},
    tokio_broadcast_channel,
};

type TaskCache = Arc<TokioMutex<HashMap<DownloadId, Arc<dyn FileDownloadTaskTrait>>>>;

#[derive(Debug, Clone)]
pub struct AsyncFetcherDownloadManager {
    manager_id: String,
    pub config: AsyncFetcherConfig,
    pub global_broadcast_sender: Arc<TokioBroadcastSender<(DownloadId, FileDownloadEvent)>>,
    tokio_handle: TokioHandle,
    task_cache: TaskCache,
}

impl AsyncFetcherDownloadManager {
    fn generate_manager_id() -> String {
        #[cfg(target_vendor = "apple")]
        {
            use crate::NSBundle;
            let bundle_id = NSBundle::mainBundle().bundleIdentifier().unwrap_or_default().to_string();

            if bundle_id.is_empty() {
                "mirai.universal".to_string()
            } else {
                format!("{}.mirai.universal", bundle_id)
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            "mirai.universal".to_string()
        }
    }

    pub async fn new_with_manager_id(
        config: AsyncFetcherConfig,
        tokio_handle: TokioHandle,
        custom_manager_id: Option<String>,
    ) -> Result<Self, DownloadError> {
        let manager_id = custom_manager_id.unwrap_or_else(|| Self::generate_manager_id());

        let (global_broadcast_sender, _) = tokio_broadcast_channel::<(DownloadId, FileDownloadEvent)>(256);
        let global_broadcast_sender = Arc::new(global_broadcast_sender);

        let task_cache: TaskCache = Arc::new(TokioMutex::new(HashMap::new()));

        let manager = Self {
            manager_id: manager_id.clone(),
            config,
            global_broadcast_sender,
            tokio_handle,
            task_cache,
        };

        tracing::debug!(
            "[DOWNLOAD_MANAGER] AsyncFetcherDownloadManager created with tokio handle, manager_id={}",
            manager_id
        );

        Ok(manager)
    }
}

#[async_trait::async_trait]
impl FileDownloadManager for AsyncFetcherDownloadManager {
    fn manager_id(&self) -> &str {
        &self.manager_id
    }

    fn subscribe_to_all_downloads(&self) -> TokioBroadcastStream<(DownloadId, FileDownloadEvent)> {
        TokioBroadcastStream::new(self.global_broadcast_sender.subscribe())
    }

    fn global_broadcast_sender(&self) -> Arc<TokioBroadcastSender<(DownloadId, FileDownloadEvent)>> {
        self.global_broadcast_sender.clone()
    }

    async fn get_all_file_tasks(&self) -> Result<Vec<Arc<dyn FileDownloadTaskTrait>>, DownloadError> {
        let task_cache_guard = self.task_cache.lock().await;
        Ok(task_cache_guard.values().cloned().collect())
    }

    async fn file_download_task(
        &self,
        source_url: &String,
        destination_path: &Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
    ) -> Result<Arc<dyn FileDownloadTaskTrait>, DownloadError> {
        use crate::managers::universal::async_fetcher_state_reducer::{
            check_crc_file_exists, check_file_exists, check_part_file_exists, reconcile_to_internal_state,
            reduce_to_checked_file_state, reduce_to_file_download_state,
        };

        let download_id = compute_download_id(source_url, destination_path);

        {
            let task_cache_guard = self.task_cache.lock().await;
            if let Some(cached_task) = task_cache_guard.get(&download_id) {
                return Ok(cached_task.clone());
            }
        }

        // Lock presence will be reflected by reduction, no error on create

        let downloaded_file_state = check_file_exists(destination_path);
        let crc_file_path = PathBuf::from(format!("{}.crc", destination_path.display()));
        let crc_file_state = check_crc_file_exists(&crc_file_path);
        let part_file_path = destination_path.with_extension("part");
        let part_file_state = check_part_file_exists(&part_file_path);

        let expected_crc = match &file_check {
            FileCheck::CRC(crc_hash) => Some(crc_hash.as_str()),
            FileCheck::None => None,
        };

        tracing::info!(
            "[MANAGER:AF] reduce_to_checked_file_state inputs: downloaded={:?}, crc={:?}, dest={}, expected_crc={:?}",
            downloaded_file_state,
            crc_file_state,
            destination_path.display(),
            expected_crc
        );
        let checked_file_state =
            reduce_to_checked_file_state(downloaded_file_state, crc_file_state, destination_path, expected_crc);

        tracing::info!(
            "[MANAGER:AF] reconcile_to_internal_state inputs: checked={:?}, part={:?}, dest={}, crc_path={}, part_path={}, expected_bytes={:?}",
            checked_file_state,
            part_file_state,
            destination_path.display(),
            crc_file_path.display(),
            part_file_path.display(),
            expected_bytes
        );
        let internal_state = reconcile_to_internal_state(
            checked_file_state,
            part_file_state,
            destination_path,
            &crc_file_path,
            &part_file_path,
            expected_bytes,
            expected_crc,
        )
        .await;

        tracing::info!(
            "[MANAGER:AF] reduce_to_file_download_state inputs: checked={:?}, part={:?}, dest={}, part_path={}, expected_bytes={:?}, manager_id={}",
            checked_file_state,
            part_file_state,
            destination_path.display(),
            part_file_path.display(),
            expected_bytes,
            self.manager_id
        );
        let file_download_state = reduce_to_file_download_state(
            checked_file_state,
            part_file_state,
            destination_path,
            &part_file_path,
            expected_bytes,
            &self.manager_id,
        );

        tracing::info!("[MANAGER:AF] initial states: internal={:?}, display={:?}", internal_state, file_download_state);

        let file_task: Arc<dyn FileDownloadTaskTrait> = Arc::new(FileDownloadTask::new(
            download_id,
            source_url.clone(),
            destination_path.to_path_buf(),
            file_check,
            self.manager_id.clone(),
            expected_bytes,
            internal_state,
            file_download_state,
            self.config.clone(),
            self.tokio_handle.clone(),
        ));

        tracing::debug!("[MANAGER] Starting listener for download_id={}", download_id);
        file_task.start_listening((*self.global_broadcast_sender).clone()).await;

        self.task_cache.lock().await.insert(download_id, file_task.clone());
        Ok(file_task)
    }
}
