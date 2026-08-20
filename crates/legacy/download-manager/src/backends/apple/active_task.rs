use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use objc2::rc::Retained;
use objc2_foundation::NSURLSessionDownloadTask;
use tokio::sync::oneshot::channel as tokio_oneshot_channel;

use crate::{
    DownloadId,
    backends::apple::{
        AppleBackend, AppleBackendError, AppleEventRegistry, AppleSinkKey, resume_data_handler::ResumeDataHandler,
        task_ext::AppleDownloadTaskExt,
    },
    traits::{ActiveTask, CancelOutcome, DownloadBackend},
};

#[derive(Debug)]
pub struct AppleActiveTask {
    task: Retained<NSURLSessionDownloadTask>,
    event_registry: AppleEventRegistry,
    sink_key: AppleSinkKey,
}

impl AppleActiveTask {
    pub fn new(
        task: Retained<NSURLSessionDownloadTask>,
        event_registry: AppleEventRegistry,
        download_id: DownloadId,
    ) -> Self {
        let sink_key = (download_id, task.task_identifier());
        Self {
            task,
            event_registry,
            sink_key,
        }
    }

    fn unregister_event_sink(&self) {
        if let Ok(mut event_registry) = self.event_registry.lock() {
            event_registry.remove(&self.sink_key);
        }
    }
}

impl Drop for AppleActiveTask {
    fn drop(&mut self) {
        self.unregister_event_sink();
        self.task.cancel();
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
        let pending_resume_data_sender = Arc::new(Mutex::new(Some(resume_data_sender)));
        self.unregister_event_sink();
        {
            let pending_resume_data_sender = Arc::clone(&pending_resume_data_sender);
            let handler = ResumeDataHandler::new_bytes(move |resume_data_bytes| {
                let resume_data_sender = match pending_resume_data_sender.lock() {
                    Ok(mut resume_data_sender) => resume_data_sender.take(),
                    Err(poisoned_sender) => {
                        let mut resume_data_sender = poisoned_sender.into_inner();
                        resume_data_sender.take()
                    },
                };
                if let Some(resume_data_sender) = resume_data_sender {
                    let _ = resume_data_sender.send(resume_data_bytes);
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
        self.unregister_event_sink();
        self.task.cancel();
        CancelOutcome::BestEffort
    }
}
