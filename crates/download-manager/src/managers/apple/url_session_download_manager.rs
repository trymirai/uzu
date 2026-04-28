use std::path::{Path, PathBuf};

use crate::{
    DownloadError, DownloadId, FileCheck, FileDownloadEvent, FileDownloadManager,
    FileDownloadTask as FileDownloadTaskTrait, compute_download_id,
    download_manager_state::DownloadManagerState,
    managers::apple::{
        FileDownloadTask, SessionConfig, URLSessionDelegate, URLSessionExt, UrlSessionDownloadTaskExt,
        url_session_state_reducer::{
            check_crc_file_exists, check_file_exists, check_resume_file_exists, reconcile_to_internal_state,
            reduce_to_checked_file_state, reduce_to_file_download_state,
        },
    },
    prelude::*,
};

#[allow(unused)]
#[derive(Debug, Clone)]
pub enum URLSessionDropPolicy {
    FinishTasksAndInvalidate,
    InvalidateAndCancel,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct URLSessionDownloadManager {
    state: DownloadManagerState,
    session: Retained<NSURLSession>,
    delegate: Retained<URLSessionDelegate>,
    delegate_protocol_object: Retained<ProtocolObject<dyn NSURLSessionDelegate>>,
    drop_policy: URLSessionDropPolicy,
}

unsafe impl Send for URLSessionDownloadManager {}
unsafe impl Sync for URLSessionDownloadManager {}

impl URLSessionDownloadManager {
    #[allow(unused)]
    pub fn manager_id(&self) -> &str {
        &self.state.manager_id
    }

    #[allow(unused)]
    pub async fn new(
        session_config: SessionConfig,
        drop_policy: URLSessionDropPolicy,
        tokio_handle: TokioHandle,
    ) -> Result<Self, DownloadError> {
        let state = DownloadManagerState::new("apple", tokio_handle);

        let backend = autoreleasepool(|_| unsafe {
            let delegate = URLSessionDelegate::new(state.global_broadcast_sender.clone());

            let delegate_protocol_object = ProtocolObject::<dyn NSURLSessionDelegate>::from_retained(delegate.clone());

            let session = NSURLSession::sessionWithConfiguration_delegate_delegateQueue(
                &session_config.ns_url_session_configuration(),
                Some(&delegate_protocol_object),
                None,
            );

            Self {
                state,
                session,
                delegate: delegate.clone(),
                delegate_protocol_object,
                drop_policy,
            }
        });

        tracing::debug!("[DOWNLOAD_MANAGER] URLSessionDownloadManager created with tokio handle");

        backend.initialize_task_cache().await?;

        tracing::debug!("[DOWNLOAD_MANAGER] Task cache initialized, ready for downloads");

        Ok(backend)
    }

    /// Invalidate the underlying NSURLSession and cancel all tasks.
    /// Safe to call multiple times; used by tests to ensure cleanup on exit.
    fn invalidate_and_cancel(&self) {
        autoreleasepool(|_| {
            self.session.invalidateAndCancel();
        });
    }

    /// Finish all outstanding tasks and invalidate the session.
    /// For background sessions, tasks may continue in the background.
    fn finish_tasks_and_invalidate(&self) {
        autoreleasepool(|_| {
            self.session.finishTasksAndInvalidate();
        });
    }

    async fn initialize_task_cache(&self) -> Result<(), DownloadError> {
        let tasks = self.session.get_tasks().await;

        for download_task in tasks.download_tasks {
            if let Some(download_info) = download_task.download_info() {
                let destination = PathBuf::from(&download_info.destination_path);
                let download_id = compute_download_id(&download_info.source_url, &destination);

                let downloaded_file_state = check_file_exists(&destination);
                let crc_file_path = PathBuf::from(format!("{}.crc", destination.display()));
                let crc_file_state = check_crc_file_exists(&crc_file_path);
                let resume_data_path = PathBuf::from(format!("{}.resume_data", destination.display()));
                let resume_data_state = check_resume_file_exists(&resume_data_path);
                let url_session_task_state = Some(download_task.state());

                let checked_file_state = reduce_to_checked_file_state(
                    downloaded_file_state,
                    crc_file_state,
                    &destination,
                    download_info.crc32c.as_deref(),
                );

                // Extract expected bytes from download task
                let expected_bytes = {
                    let count = download_task.count_of_bytes_expected_to_receive();
                    if count > 0 {
                        Some(count)
                    } else {
                        None
                    }
                };

                let internal_state = reconcile_to_internal_state(
                    checked_file_state,
                    resume_data_state,
                    Some(&download_task),
                    &destination,
                    &crc_file_path,
                    &resume_data_path,
                    expected_bytes,
                )
                .await;

                let file_download_state = reduce_to_file_download_state(
                    checked_file_state,
                    resume_data_state,
                    url_session_task_state,
                    Some(&download_task),
                    &destination,
                    expected_bytes,
                    &self.state.manager_id,
                );

                let file_check = download_info.crc32c.map_or(FileCheck::None, |crc_hash| FileCheck::CRC(crc_hash));

                let file_task = Arc::new(FileDownloadTask::new(
                    download_id,
                    download_info.source_url.clone(),
                    destination.clone(),
                    file_check,
                    self.state.manager_id.clone(),
                    expected_bytes,
                    internal_state,
                    file_download_state,
                    Some(self.session.clone()),
                    self.state.tokio_handle.clone(),
                ));

                file_task.start_listening((*self.state.global_broadcast_sender).clone()).await;

                // Special case: If task is Completed and file exists with valid CRC,
                // the delegate callback was missed (app was closed during completion).
                // Manually trigger completion handling to validate and cache CRC.
                if matches!(download_task.state(), NSURLSessionTaskState::Completed)
                    && matches!(checked_file_state, crate::CheckedFileState::Valid)
                {
                    tracing::info!(
                        "[DOWNLOAD_MANAGER] Completed task with valid file detected during init, triggering completion handling for: {}",
                        destination.display()
                    );
                    file_task.handle_download_completion().await;
                }

                self.state.task_cache.lock().await.insert(download_id, file_task);
            } else {
                download_task.cancel();
            }
        }

        Ok(())
    }
}

impl Drop for URLSessionDownloadManager {
    fn drop(&mut self) {
        match self.drop_policy {
            URLSessionDropPolicy::FinishTasksAndInvalidate => {
                self.finish_tasks_and_invalidate();
            },
            URLSessionDropPolicy::InvalidateAndCancel => {
                self.invalidate_and_cancel();
            },
        }
    }
}

#[async_trait::async_trait]
impl FileDownloadManager for URLSessionDownloadManager {
    fn manager_id(&self) -> &str {
        &self.state.manager_id
    }

    fn subscribe_to_all_downloads(&self) -> TokioBroadcastStream<(DownloadId, FileDownloadEvent)> {
        self.state.subscribe_to_all_downloads()
    }

    fn global_broadcast_sender(&self) -> Arc<TokioBroadcastSender<(DownloadId, FileDownloadEvent)>> {
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
        use crate::check_lock_file;

        let download_id = compute_download_id(source_url, destination_path);

        {
            let task_cache_guard = self.state.task_cache.lock().await;
            if let Some(cached_task) = task_cache_guard.get(&download_id) {
                return Ok(cached_task.clone());
            }
        }

        // Check lock file before proceeding (result will be used by reduction only)
        let lock_path = PathBuf::from(format!("{}.lock", destination_path.display()));
        let _ = check_lock_file(&lock_path, &self.state.manager_id, std::process::id());

        // If file is locked by another manager, do not error here.
        // Reduction will expose LockedByOther in the FileDownloadState.

        let downloaded_file_state = check_file_exists(destination_path);
        let crc_file_path = PathBuf::from(format!("{}.crc", destination_path.display()));
        let crc_file_state = check_crc_file_exists(&crc_file_path);
        let resume_data_path = PathBuf::from(format!("{}.resume_data", destination_path.display()));
        let resume_data_state = check_resume_file_exists(&resume_data_path);

        let expected_crc = match &file_check {
            FileCheck::CRC(crc_hash) => Some(crc_hash.as_str()),
            FileCheck::None => None,
        };

        let checked_file_state =
            reduce_to_checked_file_state(downloaded_file_state, crc_file_state, destination_path, expected_crc);

        let internal_state = reconcile_to_internal_state(
            checked_file_state,
            resume_data_state,
            None,
            destination_path,
            &crc_file_path,
            &resume_data_path,
            expected_bytes,
        )
        .await;

        let file_download_state = reduce_to_file_download_state(
            checked_file_state,
            resume_data_state,
            None,
            None,
            destination_path,
            expected_bytes,
            &self.state.manager_id,
        );

        // Do not acquire lock here. Lock acquisition/release is handled during
        // reconciliation when entering/leaving the Downloading state.

        let file_task = Arc::new(FileDownloadTask::new(
            download_id,
            source_url.clone(),
            destination_path.to_path_buf(),
            file_check,
            self.state.manager_id.clone(),
            expected_bytes,
            internal_state,
            file_download_state,
            Some(self.session.clone()),
            self.state.tokio_handle.clone(),
        ));

        // Start listening to global broadcast events
        tracing::debug!("[MANAGER] Starting listener for download_id={}", download_id);
        file_task.start_listening((*self.state.global_broadcast_sender).clone()).await;

        self.state.task_cache.lock().await.insert(download_id, file_task.clone());
        Ok(file_task)
    }
}
