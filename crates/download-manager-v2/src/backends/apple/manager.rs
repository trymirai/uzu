use std::{collections::HashMap, path::Path, sync::Arc};

use tokio::sync::{Mutex as TokioMutex, broadcast::channel as tokio_broadcast_channel};
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{
    DownloadError, DownloadEvent, DownloadId, FileCheck, FileDownloadManager, FileDownloadTask, FileState,
    SharedDownloadEventSender,
    backends::apple::{AppleBackend, AppleBackendContext},
    check_lock_file, compute_download_id,
    crc_utils::crc_path_for_file,
    file_download_task_actor::GenericFileDownloadTask,
    reducer::{DiskObservation, LockObservation, decide, validate},
    traits::DownloadConfig,
};

#[derive(Clone, Debug)]
pub struct AppleDownloadManager {
    manager_id: String,
    global_broadcast_sender: SharedDownloadEventSender,
    task_cache: Arc<TokioMutex<HashMap<DownloadId, Arc<dyn FileDownloadTask>>>>,
}

impl AppleDownloadManager {
    pub fn new(manager_id: String) -> Self {
        let (global_broadcast_sender, _) = tokio_broadcast_channel::<DownloadEvent>(64);
        Self {
            manager_id,
            global_broadcast_sender: Arc::new(global_broadcast_sender),
            task_cache: Arc::new(TokioMutex::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl FileDownloadManager for AppleDownloadManager {
    fn manager_id(&self) -> &str {
        &self.manager_id
    }

    fn subscribe_to_all_downloads(&self) -> TokioBroadcastStream<DownloadEvent> {
        TokioBroadcastStream::new(self.global_broadcast_sender.subscribe())
    }

    fn global_broadcast_sender(&self) -> SharedDownloadEventSender {
        Arc::clone(&self.global_broadcast_sender)
    }

    async fn get_all_file_tasks(&self) -> Result<Vec<Arc<dyn FileDownloadTask>>, DownloadError> {
        Ok(self.task_cache.lock().await.values().cloned().collect())
    }

    #[allow(clippy::ptr_arg)]
    async fn file_download_task(
        &self,
        source_url: &String,
        destination_path: &Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError> {
        let download_id = compute_download_id(source_url, destination_path);
        if let Some(cached_task) = self.task_cache.lock().await.get(&download_id) {
            return Ok(Arc::clone(cached_task));
        }

        let config = Arc::new(DownloadConfig {
            download_id,
            source_url: source_url.clone(),
            destination: destination_path.to_path_buf(),
            file_check,
            expected_bytes,
            manager_id: self.manager_id.clone(),
        });
        let resume_artifact_path = destination_path.with_extension("resume_data");
        let observation = DiskObservation {
            destination_state: file_state(destination_path),
            crc_state: FileState::Missing,
            resume_state: file_state(&resume_artifact_path),
            destination_size: file_size(destination_path),
            resume_size: file_size(&resume_artifact_path),
            expected_crc: None,
            expected_bytes,
            destination_path: destination_path.to_path_buf(),
            crc_path: Some(crc_path_for_file(destination_path)),
            resume_artifact_path: Some(resume_artifact_path),
        };
        let lock_observation = LockObservation {
            state: check_lock_file(&lock_path_for_destination(destination_path), &self.manager_id, std::process::id()),
        };
        let validation = validate(&observation);
        let decision = decide(&observation, &lock_observation, &validation);
        let task = GenericFileDownloadTask::<AppleBackend>::spawn(
            config,
            Arc::new(AppleBackendContext::default()),
            decision.initial_lifecycle_state,
            decision.initial_projection,
            decision.initial_progress,
        );

        let task: Arc<dyn FileDownloadTask> = Arc::new(task);
        self.task_cache.lock().await.insert(download_id, Arc::clone(&task));
        Ok(task)
    }
}

fn file_state(path: &Path) -> FileState {
    if path.exists() {
        FileState::Exists
    } else {
        FileState::Missing
    }
}

fn file_size(path: &Path) -> Option<u64> {
    path.metadata().ok().map(|metadata| metadata.len())
}

fn lock_path_for_destination(destination: &Path) -> std::path::PathBuf {
    std::path::PathBuf::from(format!("{}.lock", destination.display()))
}
