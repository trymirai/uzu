use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use objc2::rc::Retained;
use objc2_foundation::{NSURLSession, NSURLSessionDownloadTask};
use tokio::sync::oneshot::channel as tokio_oneshot_channel;

use crate::{
    DownloadId, RequestHeaders,
    backends::{
        apple::{
            AppleBackend, AppleBackendError, AppleEventRegistry, AppleSinkKey, DestinationInstallBarrier,
            resume_data_handler::ResumeDataHandler, task_ext::AppleDownloadTaskExt,
        },
        common::reject_symlink_components,
    },
    traits::{ActiveTask, ActiveTaskPauseOutcome, DownloadBackend},
};

pub struct AppleActiveTask {
    task: Retained<NSURLSessionDownloadTask>,
    _session: Retained<NSURLSession>,
    event_registry: AppleEventRegistry,
    sink_key: AppleSinkKey,
    resume_artifact_path: PathBuf,
    pause_resume_strategy: PauseResumeStrategy,
    destination_install_barrier: DestinationInstallBarrier,
}

impl std::fmt::Debug for AppleActiveTask {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter
            .debug_struct("AppleActiveTask")
            .field("sink_key", &self.sink_key)
            .field("resume_artifact_path", &self.resume_artifact_path)
            .field("pause_resume_strategy", &self.pause_resume_strategy)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PauseResumeStrategy {
    PersistOpaqueData,
    RestartFromBeginning,
}

impl AppleActiveTask {
    pub(crate) fn new_for_request(
        task: Retained<NSURLSessionDownloadTask>,
        session: Retained<NSURLSession>,
        event_registry: AppleEventRegistry,
        download_id: DownloadId,
        resume_artifact_path: PathBuf,
        request_headers: &RequestHeaders,
        destination_install_barrier: DestinationInstallBarrier,
    ) -> Self {
        let sink_key = AppleSinkKey::new(&session, download_id, task.task_identifier());
        let pause_resume_strategy = if request_headers.has_authorization() {
            PauseResumeStrategy::RestartFromBeginning
        } else {
            PauseResumeStrategy::PersistOpaqueData
        };
        Self {
            task,
            _session: session,
            event_registry,
            sink_key,
            resume_artifact_path,
            pause_resume_strategy,
            destination_install_barrier,
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
        self.destination_install_barrier.prevent_installation();
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
    ) -> Result<ActiveTaskPauseOutcome, <Self::Backend as DownloadBackend>::Error> {
        let resume_artifact_path = self.resume_artifact_path.clone();
        self.destination_install_barrier.prevent_installation();
        self.unregister_event_sink();
        if tokio::fs::try_exists(destination).await.map_err(|error| AppleBackendError::Io(error.to_string()))? {
            return Ok(ActiveTaskPauseOutcome::Completed);
        }

        if self.pause_resume_strategy == PauseResumeStrategy::RestartFromBeginning {
            self.task.cancel();
            write_resume_data(&resume_artifact_path, &[]).await?;
            return Ok(ActiveTaskPauseOutcome::Paused(resume_artifact_path));
        }

        let (resume_data_sender, resume_data_receiver) = tokio_oneshot_channel::<Box<[u8]>>();
        let pending_resume_data_sender = Arc::new(Mutex::new(Some(resume_data_sender)));
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
        write_resume_data(&resume_artifact_path, &resume_data_bytes).await?;
        Ok(ActiveTaskPauseOutcome::Paused(resume_artifact_path))
    }

    async fn cancel(
        self,
        _destination: &Path,
    ) {
        self.destination_install_barrier.prevent_installation();
        self.unregister_event_sink();
        self.task.cancel();
    }
}

pub(super) async fn write_resume_data(
    path: &Path,
    bytes: &[u8],
) -> Result<(), AppleBackendError> {
    reject_symlink_components(path).await.map_err(|error| AppleBackendError::Io(error.to_string()))?;
    tokio::fs::write(path, bytes).await.map_err(|error| AppleBackendError::Io(error.to_string()))
}

#[cfg(test)]
#[path = "../../../tests/unit/backends/apple/active_task_test.rs"]
mod tests;
