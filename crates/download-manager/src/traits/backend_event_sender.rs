use std::sync::Arc;

use tokio::sync::{Mutex as TokioMutex, mpsc::Sender as TokioMpscSender, watch::Sender as TokioWatchSender};

use crate::{
    DownloadId,
    download_log_event::{DownloadLogEvent, log},
    file_download_task_actor::{BackendEvent, BackendProgress, PendingProgressSlot},
    traits::ActiveDownloadGeneration,
};

#[derive(Clone, Debug)]
pub struct BackendEventSender {
    download_id: DownloadId,
    terminal_event_sender: TokioMpscSender<BackendEvent>,
    progress_coalescer: SharedProgressCoalescer,
}

impl BackendEventSender {
    pub fn new(
        download_id: DownloadId,
        terminal_event_sender: TokioMpscSender<BackendEvent>,
        pending_progress: Arc<TokioMutex<PendingProgressSlot>>,
        actor_waker: TokioWatchSender<()>,
    ) -> Self {
        Self {
            download_id,
            terminal_event_sender,
            progress_coalescer: SharedProgressCoalescer {
                pending_progress,
                actor_waker,
            },
        }
    }

    pub async fn send_terminal(
        &self,
        event: BackendEvent,
    ) -> Result<(), BackendEvent> {
        log(DownloadLogEvent::BackendTerminal {
            download_id: self.download_id,
            event: event.clone(),
        });
        self.terminal_event_sender.send(event).await.map_err(|error| error.0)
    }

    pub async fn send_progress(
        &self,
        generation: ActiveDownloadGeneration,
        downloaded_bytes: u64,
        total_bytes: Option<u64>,
    ) {
        log(DownloadLogEvent::BackendProgress {
            download_id: self.download_id,
            generation,
            downloaded_bytes,
            total_bytes,
        });
        let mut pending_progress = self.progress_coalescer.pending_progress.lock().await;
        pending_progress.progress = Some(BackendProgress {
            generation,
            downloaded_bytes,
            total_bytes,
        });
        let _ = self.progress_coalescer.actor_waker.send(());
    }
}

#[derive(Clone, Debug)]
struct SharedProgressCoalescer {
    pending_progress: Arc<TokioMutex<PendingProgressSlot>>,
    actor_waker: TokioWatchSender<()>,
}
