use tokio::sync::{mpsc::Sender as TokioMpscSender, watch::Sender as TokioWatchSender};

use crate::{
    DownloadId,
    backends::{BackendEvent, BackendProgress, DownloadGeneration},
};

#[derive(Clone, Debug)]
pub struct BackendEventSender {
    download_id: DownloadId,
    terminal: TokioMpscSender<BackendEvent>,
    progress: TokioWatchSender<Option<BackendProgress>>,
}

impl BackendEventSender {
    pub fn new(
        download_id: DownloadId,
        terminal: TokioMpscSender<BackendEvent>,
        progress: TokioWatchSender<Option<BackendProgress>>,
    ) -> Self {
        Self {
            download_id,
            terminal,
            progress,
        }
    }

    pub async fn send_terminal(
        &self,
        event: BackendEvent,
    ) {
        tracing::debug!(download_id = %self.download_id, ?event, "backend terminal event");
        if let Err(dropped) = self.terminal.send(event).await {
            tracing::debug!(download_id = %self.download_id, event = ?dropped.0, "actor gone, terminal event dropped");
        }
    }

    pub fn send_progress(
        &self,
        generation: DownloadGeneration,
        downloaded_bytes: u64,
        total_bytes: Option<u64>,
    ) {
        tracing::trace!(download_id = %self.download_id, ?generation, downloaded_bytes, total_bytes, "backend progress");
        self.progress.send_replace(Some(BackendProgress {
            generation,
            downloaded_bytes,
            total_bytes,
        }));
    }
}
