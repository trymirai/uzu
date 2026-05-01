use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{
    Arc, DownloadError, DownloadEvent, FileCheck, FileDownloadManager, FileDownloadTask as FileDownloadTaskTrait, Path,
    PathBuf, SharedDownloadEventSender, TokioHandle, compute_download_id,
    download_manager_state::DownloadManagerState,
    managers::universal::{AsyncFetcherConfig, FileDownloadTask},
};

#[derive(Debug, Clone)]
pub struct AsyncFetcherDownloadManager {
    state: DownloadManagerState,
    pub config: AsyncFetcherConfig,
}

impl AsyncFetcherDownloadManager {
    #[allow(unused)]
    pub fn manager_id(&self) -> &str {
        &self.state.manager_id
    }

    #[allow(unused)]
    pub async fn new(
        config: AsyncFetcherConfig,
        tokio_handle: TokioHandle,
    ) -> Result<Self, DownloadError> {
        let state = DownloadManagerState::new("universal", tokio_handle);

        let manager = Self {
            state,
            config,
        };

        tracing::debug!(
            "[DOWNLOAD_MANAGER] AsyncFetcherDownloadManager created with tokio handle, manager_id={}",
            manager.manager_id()
        );

        Ok(manager)
    }
}

#[async_trait::async_trait]
impl FileDownloadManager for AsyncFetcherDownloadManager {
    #[allow(unused)]
    fn manager_id(&self) -> &str {
        &self.state.manager_id
    }

    fn subscribe_to_all_downloads(&self) -> TokioBroadcastStream<DownloadEvent> {
        self.state.subscribe_to_all_downloads()
    }

    fn global_broadcast_sender(&self) -> SharedDownloadEventSender {
        self.state.global_broadcast_sender.clone()
    }

    async fn get_all_file_tasks(&self) -> Result<Vec<Arc<dyn FileDownloadTaskTrait>>, DownloadError> {
        self.state.get_all_file_tasks().await
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
            let task_cache_guard = self.state.task_cache.lock().await;
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
            self.state.manager_id
        );
        let file_download_state = reduce_to_file_download_state(
            checked_file_state,
            part_file_state,
            destination_path,
            &part_file_path,
            expected_bytes,
            &self.state.manager_id,
        );

        tracing::info!("[MANAGER:AF] initial states: internal={:?}, display={:?}", internal_state, file_download_state);

        let file_task: Arc<dyn FileDownloadTaskTrait> = Arc::new(FileDownloadTask::new(
            download_id,
            source_url.clone(),
            destination_path.to_path_buf(),
            file_check,
            self.state.manager_id.clone(),
            expected_bytes,
            internal_state,
            file_download_state,
            self.config.clone(),
            self.state.tokio_handle.clone(),
        ));

        tracing::debug!("[MANAGER] Starting listener for download_id={}", download_id);
        file_task.start_listening((*self.state.global_broadcast_sender).clone()).await;

        self.state.task_cache.lock().await.insert(download_id, file_task.clone());
        Ok(file_task)
    }
}
