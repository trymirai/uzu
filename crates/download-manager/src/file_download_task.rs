use std::{fmt::Debug, path::Path};

use tokio::sync::broadcast::Sender as TokioBroadcastSender;
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{DownloadError, DownloadEventSender, DownloadId, FileCheck, FileDownloadState};

#[async_trait::async_trait]
pub trait FileDownloadTask: Send + Sync + Debug {
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
    async fn wait(&self);

    fn broadcast_sender(&self) -> TokioBroadcastSender<FileDownloadState>;
}
