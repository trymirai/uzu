#![allow(dead_code)]

use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use download_manager::{
    DownloadError, DownloadEvent, DownloadEventSender, DownloadId, FileCheck, FileDownloadManager, FileDownloadState,
    FileDownloadTask, SharedDownloadEventSender,
};
use tokio::sync::{
    Mutex as TokioMutex,
    broadcast::{Sender as TokioBroadcastSender, channel as tokio_broadcast_channel},
};
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;
use uuid::Uuid;

#[derive(Debug)]
pub struct FailingFileTask {
    download_id: DownloadId,
    source_url: String,
    destination: PathBuf,
    state: TokioMutex<FileDownloadState>,
    file_check: FileCheck,
    download_response: Option<Result<(), DownloadError>>,
    pause_response: Option<Result<(), DownloadError>>,
    cancel_response: Option<Result<(), DownloadError>>,
}

impl FailingFileTask {
    pub fn new(
        source_url: String,
        destination: PathBuf,
        initial_state: FileDownloadState,
    ) -> Self {
        Self {
            download_id: Uuid::new_v4(),
            source_url,
            destination,
            state: TokioMutex::new(initial_state),
            file_check: FileCheck::None,
            download_response: None,
            pause_response: None,
            cancel_response: None,
        }
    }

    pub fn with_download_error(
        mut self,
        message: impl Into<String>,
    ) -> Self {
        self.download_response = Some(Err(DownloadError::Backend(message.into())));
        self
    }

    pub fn with_pause_error(
        mut self,
        message: impl Into<String>,
    ) -> Self {
        self.pause_response = Some(Err(DownloadError::Backend(message.into())));
        self
    }

    pub fn with_cancel_error(
        mut self,
        message: impl Into<String>,
    ) -> Self {
        self.cancel_response = Some(Err(DownloadError::Backend(message.into())));
        self
    }
}

#[async_trait]
impl FileDownloadTask for FailingFileTask {
    fn download_id(&self) -> DownloadId {
        self.download_id
    }
    fn source_url(&self) -> &str {
        &self.source_url
    }
    fn destination(&self) -> &std::path::Path {
        &self.destination
    }
    fn file_check(&self) -> &FileCheck {
        &self.file_check
    }
    async fn download(&self) -> Result<(), DownloadError> {
        match &self.download_response {
            Some(response) => response.clone(),
            None => unreachable!("test did not expect file_task.download() to be invoked"),
        }
    }
    async fn pause(&self) -> Result<(), DownloadError> {
        match &self.pause_response {
            Some(response) => response.clone(),
            None => unreachable!("test did not expect file_task.pause() to be invoked"),
        }
    }
    async fn cancel(&self) -> Result<(), DownloadError> {
        match &self.cancel_response {
            Some(response) => response.clone(),
            None => unreachable!("test did not expect file_task.cancel() to be invoked"),
        }
    }
    async fn state(&self) -> FileDownloadState {
        self.state.lock().await.clone()
    }
    async fn progress(&self) -> Result<TokioBroadcastStream<FileDownloadState>, DownloadError> {
        unreachable!("test did not expect file_task.progress() to be invoked");
    }
    async fn start_listening(
        &self,
        _: DownloadEventSender,
    ) {
        unreachable!("test did not expect file_task.start_listening() to be invoked");
    }
    async fn stop_listening(&self) {
        unreachable!("test did not expect file_task.stop_listening() to be invoked");
    }
    async fn wait(&self) {
        unreachable!("test did not expect file_task.wait() to be invoked");
    }
    fn broadcast_sender(&self) -> TokioBroadcastSender<FileDownloadState> {
        unreachable!("test did not expect file_task.broadcast_sender() to be invoked");
    }
}

#[derive(Debug)]
pub struct StubManager {
    sender: TokioBroadcastSender<DownloadEvent>,
}

impl StubManager {
    pub fn new() -> Self {
        let (sender, _) = tokio_broadcast_channel(1);
        Self {
            sender,
        }
    }
}

#[async_trait]
impl FileDownloadManager for StubManager {
    fn manager_id(&self) -> &str {
        "stub"
    }
    fn subscribe_to_all_downloads(&self) -> TokioBroadcastStream<DownloadEvent> {
        TokioBroadcastStream::new(self.sender.subscribe())
    }
    fn global_broadcast_sender(&self) -> SharedDownloadEventSender {
        Arc::new(self.sender.clone())
    }
    async fn get_all_file_tasks(&self) -> Result<Vec<Arc<dyn FileDownloadTask>>, DownloadError> {
        Ok(Vec::new())
    }
    async fn remove_file_task(
        &self,
        _: DownloadId,
    ) -> Result<(), DownloadError> {
        Ok(())
    }
    async fn file_download_task(
        &self,
        _: &String,
        _: &std::path::Path,
        _: FileCheck,
        _: Option<u64>,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError> {
        unreachable!("test did not expect manager.file_download_task() to be invoked");
    }
}
