use std::sync::Arc;

use objc2_foundation::NSURLSessionTaskState;

use crate::{
    DownloadError,
    backends::{
        apple::{AppleActiveTask, AppleBackendContext, AppleBackendError, task_ext::AppleDownloadTaskExt},
        common::{self, InitialTaskAttachment},
    },
    traits::{ActiveDownloadGeneration, BackendEventSender, DownloadBackend, DownloadConfig},
};

#[derive(Clone, Debug, Default)]
pub struct AppleBackend;

impl DownloadBackend for AppleBackend {
    type Context = AppleBackendContext;
    type ActiveTask = AppleActiveTask;
    type Error = AppleBackendError;
}

#[async_trait::async_trait]
impl common::Backend for AppleBackend {
    const RESUME_ARTIFACT_EXTENSION: &'static str = "resume_data";

    fn manager_suffix() -> &'static str {
        "apple"
    }

    fn create_context(tokio_handle: tokio::runtime::Handle) -> Result<Self::Context, DownloadError> {
        Ok(AppleBackendContext::new(tokio_handle))
    }

    async fn initial_task_attachment(
        context: &Self::Context,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        backend_event_sender: BackendEventSender,
    ) -> Result<InitialTaskAttachment<Self>, DownloadError> {
        let Some(task) = context.matching_download_task(&config) else {
            return Ok(InitialTaskAttachment::None);
        };

        match task.state() {
            NSURLSessionTaskState::Running | NSURLSessionTaskState::Suspended => {
                let initial_downloaded_bytes = task.count_of_bytes_received();
                let total_bytes = match task.count_of_bytes_expected_to_receive() {
                    0 => config.expected_bytes,
                    total_bytes => Some(total_bytes),
                };
                context.attach_existing_task(&task, Arc::clone(&config), generation, backend_event_sender);
                if matches!(task.state(), NSURLSessionTaskState::Suspended) {
                    task.resume();
                }
                Ok(InitialTaskAttachment::Downloading {
                    active_task: AppleActiveTask::wrap(task),
                    initial_downloaded_bytes,
                    total_bytes,
                })
            },
            NSURLSessionTaskState::Completed | NSURLSessionTaskState::Canceling => {
                task.cancel();
                Ok(InitialTaskAttachment::None)
            },
            _ => Ok(InitialTaskAttachment::None),
        }
    }
}
