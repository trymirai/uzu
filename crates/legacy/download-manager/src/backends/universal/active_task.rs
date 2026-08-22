use std::path::{Path, PathBuf};

use kiban::rt::TaskJoinHandle;
use tokio::sync::{oneshot::Receiver as TokioOneshotReceiver, watch::Sender as TokioWatchSender};

use crate::{
    backends::universal::UniversalBackend,
    traits::{ActiveTask, ActiveTaskPauseOutcome, DownloadBackend},
};

pub struct UniversalActiveTask {
    task_handles: Box<[Box<dyn TaskJoinHandle<()>>]>,
    pause_sender: Option<TokioWatchSender<bool>>,
    completion_receiver: Option<TokioOneshotReceiver<ActiveTaskPauseOutcome>>,
    resume_artifact_path: PathBuf,
}

impl std::fmt::Debug for UniversalActiveTask {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter
            .debug_struct("UniversalActiveTask")
            .field("resume_artifact_path", &self.resume_artifact_path)
            .finish_non_exhaustive()
    }
}

impl UniversalActiveTask {
    pub fn new(
        task_handles: Box<[Box<dyn TaskJoinHandle<()>>]>,
        pause_sender: TokioWatchSender<bool>,
        completion_receiver: TokioOneshotReceiver<ActiveTaskPauseOutcome>,
        resume_artifact_path: PathBuf,
    ) -> Self {
        Self {
            task_handles,
            pause_sender: Some(pause_sender),
            completion_receiver: Some(completion_receiver),
            resume_artifact_path,
        }
    }
}

impl Drop for UniversalActiveTask {
    fn drop(&mut self) {
        for handle in self.task_handles.iter() {
            handle.abort();
        }
    }
}

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
impl ActiveTask for UniversalActiveTask {
    type Backend = UniversalBackend;

    async fn pause(
        mut self,
        _destination: &Path,
    ) -> Result<ActiveTaskPauseOutcome, <Self::Backend as DownloadBackend>::Error> {
        if let Some(pause_sender) = self.pause_sender.take() {
            let _ = pause_sender.send(true);
        }

        let outcome = match self.completion_receiver.take() {
            Some(receiver) => receiver.await,
            None => {
                return Err(crate::backends::universal::UniversalBackendError::Io(
                    "download worker pause outcome was already consumed".to_string(),
                ));
            },
        };
        match outcome {
            Ok(ActiveTaskPauseOutcome::Paused(_)) => {
                let _ = std::mem::take(&mut self.task_handles);
                Ok(ActiveTaskPauseOutcome::Paused(std::mem::take(&mut self.resume_artifact_path)))
            },
            Ok(outcome) => {
                let _ = std::mem::take(&mut self.task_handles);
                Ok(outcome)
            },
            Err(_) => {
                for task_handle in std::mem::take(&mut self.task_handles) {
                    task_handle.abort_and_join().await;
                }
                Err(crate::backends::universal::UniversalBackendError::Io(
                    "download worker stopped before reporting its pause outcome".to_string(),
                ))
            },
        }
    }

    async fn cancel(
        mut self,
        _destination: &Path,
    ) {
        for task_handle in std::mem::take(&mut self.task_handles) {
            task_handle.abort_and_join().await;
        }
    }
}
