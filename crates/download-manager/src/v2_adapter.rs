use std::{collections::HashMap, path::Path, sync::Arc};

use tokio::{
    sync::{
        Mutex as TokioMutex,
        broadcast::{Sender as TokioBroadcastSender, channel as tokio_broadcast_channel},
    },
    task::JoinHandle as TokioJoinHandle,
};
use tokio_stream::{StreamExt, wrappers::BroadcastStream as TokioBroadcastStream};

use crate::{
    DownloadError, DownloadEvent, DownloadEventSender, DownloadId, FileCheck, FileDownloadEvent, FileDownloadManager,
    FileDownloadManagerType, FileDownloadPhase, FileDownloadState, FileDownloadTask, SharedDownloadEventSender,
    compute_download_id,
};

pub(crate) struct V2DownloadManagerAdapter {
    inner: Box<dyn download_manager_v2::FileDownloadManager>,
    task_cache: TokioMutex<HashMap<DownloadId, Arc<dyn FileDownloadTask>>>,
    global_broadcast_sender: SharedDownloadEventSender,
}

impl V2DownloadManagerAdapter {
    pub fn new(inner: Box<dyn download_manager_v2::FileDownloadManager>) -> Self {
        let (global_broadcast_sender, _) = tokio_broadcast_channel::<DownloadEvent>(64);
        Self {
            inner,
            task_cache: TokioMutex::new(HashMap::new()),
            global_broadcast_sender: Arc::new(global_broadcast_sender),
        }
    }
}

#[async_trait::async_trait]
impl FileDownloadManager for V2DownloadManagerAdapter {
    fn manager_id(&self) -> &str {
        self.inner.manager_id()
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

    async fn file_download_task(
        &self,
        source_url: &String,
        destination_path: &Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError> {
        let download_id = compute_download_id(source_url, destination_path);
        if let Some(task) = self.task_cache.lock().await.get(&download_id) {
            return Ok(Arc::clone(task));
        }

        let v2_task = self
            .inner
            .file_download_task(source_url, destination_path, to_v2_file_check(file_check), expected_bytes)
            .await
            .map_err(from_v2_error)?;
        let task: Arc<dyn FileDownloadTask> = Arc::new(V2FileDownloadTaskAdapter::new(v2_task));
        task.start_listening((*self.global_broadcast_sender).clone()).await;
        self.task_cache.lock().await.insert(download_id, Arc::clone(&task));
        Ok(task)
    }
}

struct V2FileDownloadTaskAdapter {
    inner: Arc<dyn download_manager_v2::FileDownloadTask>,
    file_check: FileCheck,
    broadcast_sender: TokioBroadcastSender<FileDownloadState>,
    local_forward_task: TokioMutex<Option<TokioJoinHandle<()>>>,
    global_forward_task: TokioMutex<Option<TokioJoinHandle<()>>>,
}

impl V2FileDownloadTaskAdapter {
    fn new(inner: Arc<dyn download_manager_v2::FileDownloadTask>) -> Self {
        let (broadcast_sender, _) = tokio_broadcast_channel(64);
        let file_check = from_v2_file_check(inner.file_check().clone());
        Self {
            inner,
            file_check,
            broadcast_sender,
            local_forward_task: TokioMutex::new(None),
            global_forward_task: TokioMutex::new(None),
        }
    }

    async fn ensure_local_forwarder(&self) -> Result<(), DownloadError> {
        let mut local_forward_task = self.local_forward_task.lock().await;
        if local_forward_task.is_some() {
            return Ok(());
        }

        let mut v2_progress = self.inner.progress().await.map_err(from_v2_error)?;
        let broadcast_sender = self.broadcast_sender.clone();
        *local_forward_task = Some(tokio::spawn(async move {
            while let Some(result) = v2_progress.next().await {
                let Ok(v2_state) = result else {
                    continue;
                };
                let _ = broadcast_sender.send(from_v2_file_download_state(v2_state));
            }
        }));
        Ok(())
    }
}

impl std::fmt::Debug for V2FileDownloadTaskAdapter {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter
            .debug_struct("V2FileDownloadTaskAdapter")
            .field("download_id", &self.inner.download_id())
            .field("source_url", &self.inner.source_url())
            .field("destination", &self.inner.destination())
            .finish()
    }
}

#[async_trait::async_trait]
impl FileDownloadTask for V2FileDownloadTaskAdapter {
    fn download_id(&self) -> DownloadId {
        self.inner.download_id()
    }

    fn source_url(&self) -> &str {
        self.inner.source_url()
    }

    fn destination(&self) -> &Path {
        self.inner.destination()
    }

    fn file_check(&self) -> &FileCheck {
        &self.file_check
    }

    async fn download(&self) -> Result<(), DownloadError> {
        self.ensure_local_forwarder().await?;
        self.inner.download().await.map_err(from_v2_error)
    }

    async fn pause(&self) -> Result<(), DownloadError> {
        self.inner.pause().await.map_err(from_v2_error)
    }

    async fn cancel(&self) -> Result<(), DownloadError> {
        self.inner.cancel().await.map_err(from_v2_error)
    }

    async fn state(&self) -> FileDownloadState {
        from_v2_file_download_state(self.inner.state().await)
    }

    async fn progress(&self) -> Result<TokioBroadcastStream<FileDownloadState>, DownloadError> {
        self.ensure_local_forwarder().await?;
        Ok(TokioBroadcastStream::new(self.broadcast_sender.subscribe()))
    }

    async fn start_listening(
        &self,
        global_broadcast: DownloadEventSender,
    ) {
        let _ = self.ensure_local_forwarder().await;
        let mut global_forward_task = self.global_forward_task.lock().await;
        if global_forward_task.is_some() {
            return;
        }

        let download_id = self.inner.download_id();
        let destination = self.inner.destination().to_path_buf();
        let mut local_progress = TokioBroadcastStream::new(self.broadcast_sender.subscribe());
        *global_forward_task = Some(tokio::spawn(async move {
            let mut last_downloaded_bytes = 0_u64;
            while let Some(result) = local_progress.next().await {
                let Ok(state) = result else {
                    continue;
                };

                match state.phase {
                    FileDownloadPhase::Downloading => {
                        let bytes_written = state.downloaded_bytes.saturating_sub(last_downloaded_bytes);
                        last_downloaded_bytes = state.downloaded_bytes;
                        let _ = global_broadcast.send((
                            download_id,
                            FileDownloadEvent::ProgressUpdate {
                                bytes_written,
                                total_bytes_written: state.downloaded_bytes,
                                total_bytes_expected: state.total_bytes,
                            },
                        ));
                    },
                    FileDownloadPhase::Downloaded => {
                        let _ = global_broadcast.send((
                            download_id,
                            FileDownloadEvent::DownloadCompleted {
                                tmp_path: destination.clone(),
                                final_destination: destination.clone(),
                            },
                        ));
                        break;
                    },
                    FileDownloadPhase::Error(message) => {
                        let _ = global_broadcast.send((
                            download_id,
                            FileDownloadEvent::Error {
                                message,
                            },
                        ));
                        break;
                    },
                    FileDownloadPhase::NotDownloaded
                    | FileDownloadPhase::Paused
                    | FileDownloadPhase::LockedByOther(_) => {},
                }
            }
        }));
    }

    async fn stop_listening(&self) {
        if let Some(task) = self.global_forward_task.lock().await.take() {
            task.abort();
            let _ = task.await;
        }
    }

    async fn wait(&self) {
        self.inner.wait().await;
    }

    fn broadcast_sender(&self) -> TokioBroadcastSender<FileDownloadState> {
        self.broadcast_sender.clone()
    }
}

fn to_v2_file_check(file_check: FileCheck) -> download_manager_v2::FileCheck {
    match file_check {
        FileCheck::CRC(crc) => download_manager_v2::FileCheck::CRC(crc),
        FileCheck::None => download_manager_v2::FileCheck::None,
    }
}

fn from_v2_file_check(file_check: download_manager_v2::FileCheck) -> FileCheck {
    match file_check {
        download_manager_v2::FileCheck::CRC(crc) => FileCheck::CRC(crc),
        download_manager_v2::FileCheck::None => FileCheck::None,
    }
}

fn from_v2_file_download_state(state: download_manager_v2::FileDownloadState) -> FileDownloadState {
    FileDownloadState {
        total_bytes: state.total_bytes,
        downloaded_bytes: state.downloaded_bytes,
        phase: from_v2_file_download_phase(state.phase),
    }
}

fn from_v2_file_download_phase(phase: download_manager_v2::FileDownloadPhase) -> FileDownloadPhase {
    match phase {
        download_manager_v2::FileDownloadPhase::NotDownloaded => FileDownloadPhase::NotDownloaded,
        download_manager_v2::FileDownloadPhase::Downloading => FileDownloadPhase::Downloading,
        download_manager_v2::FileDownloadPhase::Paused => FileDownloadPhase::Paused,
        download_manager_v2::FileDownloadPhase::Downloaded => FileDownloadPhase::Downloaded,
        download_manager_v2::FileDownloadPhase::LockedByOther(manager_id) => {
            FileDownloadPhase::LockedByOther(manager_id)
        },
        download_manager_v2::FileDownloadPhase::Error(message) => FileDownloadPhase::Error(message),
    }
}

fn from_v2_error(error: download_manager_v2::DownloadError) -> DownloadError {
    match error {
        download_manager_v2::DownloadError::HttpStatus(status) => DownloadError::HttpStatus(status),
        download_manager_v2::DownloadError::Canceled => DownloadError::Canceled,
        download_manager_v2::DownloadError::ResumeUnsupported => DownloadError::ResumeUnsupported,
        download_manager_v2::DownloadError::UnsupportedType => DownloadError::UnsupportedType,
        download_manager_v2::DownloadError::BadUrl => DownloadError::BadUrl,
        download_manager_v2::DownloadError::MissingDownloadInfo => DownloadError::MissingDownloadInfo,
        download_manager_v2::DownloadError::ResumeDataReadFailed => DownloadError::ResumeDataReadFailed,
        download_manager_v2::DownloadError::ResumeDataError => DownloadError::ResumeDataError,
        download_manager_v2::DownloadError::DownloadTaskNotFound => DownloadError::DownloadTaskNotFound,
        download_manager_v2::DownloadError::TaskNotFoundAfterCreation => DownloadError::TaskNotFoundAfterCreation,
        download_manager_v2::DownloadError::NoMatchingTaskToPause => DownloadError::NoMatchingTaskToPause,
        download_manager_v2::DownloadError::UnknownDownloadHandle => DownloadError::UnknownDownloadHandle,
        download_manager_v2::DownloadError::MutexPoisoned => DownloadError::MutexPoisoned,
        download_manager_v2::DownloadError::InvalidStateTransition => DownloadError::InvalidStateTransition,
        download_manager_v2::DownloadError::LockedByOther(manager_id) => DownloadError::LockedByOther(manager_id),
        download_manager_v2::DownloadError::Io(message)
        | download_manager_v2::DownloadError::SerdeJson(message)
        | download_manager_v2::DownloadError::Backend(message) => DownloadError::IOError(message),
        download_manager_v2::DownloadError::TaskStopped => DownloadError::IOError("task stopped".to_string()),
        download_manager_v2::DownloadError::ChannelClosed => DownloadError::IOError("channel closed".to_string()),
    }
}

impl From<FileDownloadManagerType> for download_manager_v2::FileDownloadManagerType {
    fn from(file_download_manager_type: FileDownloadManagerType) -> Self {
        match file_download_manager_type {
            FileDownloadManagerType::Universal => Self::Universal,
            FileDownloadManagerType::Apple => Self::Apple,
        }
    }
}
