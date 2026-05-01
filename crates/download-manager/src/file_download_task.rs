use std::path::Path;

use crate::{
    DownloadError, DownloadEventSender, DownloadId, FileCheck, FileDownloadState,
    prelude::{TokioBroadcastSender, TokioBroadcastStream},
};

#[async_trait::async_trait]
pub trait FileDownloadTask: Send + Sync + std::fmt::Debug {
    fn download_id(&self) -> DownloadId;
    fn source_url(&self) -> &str;
    fn destination(&self) -> &Path;
    fn file_check(&self) -> &FileCheck;

    async fn download(&self) -> Result<(), DownloadError>;
    async fn pause(&self) -> Result<(), DownloadError>;
    async fn cancel(&self) -> Result<(), DownloadError>;
    async fn state(&self) -> FileDownloadState;
    async fn progress(&self) -> Result<TokioBroadcastStream<FileDownloadState>, DownloadError>;

    async fn start_listening(
        &self,
        global_broadcast: DownloadEventSender,
    );
    async fn stop_listening(&self);

    /// Wait for the task to fully complete (download finished, lock released, state updated).
    async fn wait(&self);

    fn broadcast_sender(&self) -> TokioBroadcastSender<FileDownloadState>;
}
