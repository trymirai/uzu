use std::path::Path;

use kiban::rt::TaskJoinHandle;
use tokio::sync::{oneshot::Receiver as TokioOneshotReceiver, watch::Sender as TokioWatchSender};

use crate::{DownloadError, backends::ActiveTask};

pub struct UniversalActiveTask {
    task_handle: Box<dyn TaskJoinHandle<()>>,
    pause: TokioWatchSender<bool>,
    completion: TokioOneshotReceiver<()>,
}

impl UniversalActiveTask {
    pub fn new(
        task_handle: Box<dyn TaskJoinHandle<()>>,
        pause: TokioWatchSender<bool>,
        completion: TokioOneshotReceiver<()>,
    ) -> Self {
        Self {
            task_handle,
            pause,
            completion,
        }
    }
}

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
impl ActiveTask for UniversalActiveTask {
    async fn pause(
        self: Box<Self>,
        _resume_artifact_path: &Path,
    ) -> Result<(), DownloadError> {
        let _ = self.pause.send(true);
        let _ = self.completion.await;
        self.task_handle.abort_and_join().await;
        Ok(())
    }

    async fn cancel(self: Box<Self>) {
        self.task_handle.abort_and_join().await;
    }
}
