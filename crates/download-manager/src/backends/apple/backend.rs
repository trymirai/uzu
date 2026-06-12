use std::{path::Path, sync::Arc};

use objc2_foundation::NSURLSessionTaskState;

use crate::{
    DownloadError,
    backends::{
        apple::{
            AppleActiveTask, AppleBackendContext, AppleBackendError, resume_data_parser, task_ext::AppleDownloadTaskExt,
        },
        common::{self, InitialTaskAttachment},
    },
    lock_manager::DestinationLockLease,
    traits::{ActiveDownloadGeneration, BackendEventSender, DownloadBackend, DownloadConfig},
};

#[derive(Clone, Debug, Default)]
pub struct AppleBackend;

impl DownloadBackend for AppleBackend {
    type Context = AppleBackendContext;
    type ActiveTask = AppleActiveTask;
    type Error = AppleBackendError;

    fn read_resume_progress(part_path: &Path) -> Option<u64> {
        resume_data_parser::read_resume_progress(part_path)
    }
}

#[async_trait::async_trait]
impl common::Backend for AppleBackend {
    const RESUME_ARTIFACT_EXTENSION: &'static str = "resume_data";
    const SUPPORTS_INITIAL_TASK_ATTACHMENT: bool = true;

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
        _destination_lease: &DestinationLockLease,
    ) -> Result<InitialTaskAttachment<Self>, DownloadError> {
        let Some(task) = context
            .claim_matching_download_task(&config)
            .await
            .map_err(|error| DownloadError::Backend(error.to_string()))?
        else {
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
                    active_task: AppleActiveTask::new(task, context.event_registry(), config.download_id),
                    initial_downloaded_bytes,
                    total_bytes,
                })
            },
            NSURLSessionTaskState::Completed | NSURLSessionTaskState::Canceling => Ok(InitialTaskAttachment::None),
            _ => Ok(InitialTaskAttachment::None),
        }
    }

    async fn has_initial_task_to_claim(
        context: &Self::Context,
        config: &DownloadConfig,
    ) -> Result<bool, DownloadError> {
        context.has_download_task_to_claim(config).await.map_err(|error| DownloadError::Backend(error.to_string()))
    }
}
