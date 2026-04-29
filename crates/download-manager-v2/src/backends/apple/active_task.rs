use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use objc2::rc::Retained;
use objc2_foundation::NSURLSessionDownloadTask;
use tokio::sync::oneshot::channel as tokio_oneshot_channel;

use crate::{
    backends::apple::{AppleBackend, AppleBackendError, resume_data_handler::ResumeDataHandler},
    traits::{ActiveTask, CancelOutcome, DownloadBackend},
};

#[derive(Debug)]
pub struct AppleActiveTask {
    task: Retained<NSURLSessionDownloadTask>,
}

impl AppleActiveTask {
    pub fn wrap(task: Retained<NSURLSessionDownloadTask>) -> Self {
        Self {
            task,
        }
    }
}

#[async_trait::async_trait]
impl ActiveTask for AppleActiveTask {
    type Backend = AppleBackend;

    async fn pause(
        self,
        destination: &Path,
    ) -> Result<PathBuf, <Self::Backend as DownloadBackend>::Error> {
        let resume_artifact_path = destination.with_extension("resume_data");
        let (resume_data_sender, resume_data_receiver) = tokio_oneshot_channel::<Box<[u8]>>();
        let resume_data_sender = Arc::new(Mutex::new(Some(resume_data_sender)));
        {
            let resume_data_sender = Arc::clone(&resume_data_sender);
            let handler = ResumeDataHandler::new_bytes(move |resume_data_bytes| {
                if let Some(sender) = resume_data_sender.lock().ok().and_then(|mut sender| sender.take()) {
                    let _ = sender.send(resume_data_bytes);
                }
            });
            unsafe {
                self.task.cancelByProducingResumeData(&handler);
            }
        }
        let resume_data_bytes =
            resume_data_receiver.await.map_err(|error| AppleBackendError::ResumeData(error.to_string()))?;
        tokio::fs::write(&resume_artifact_path, resume_data_bytes)
            .await
            .map_err(|error| AppleBackendError::Io(error.to_string()))?;
        Ok(resume_artifact_path)
    }

    async fn cancel(
        self,
        _destination: &Path,
    ) -> CancelOutcome {
        self.task.cancel();
        CancelOutcome::BestEffort
    }
}
