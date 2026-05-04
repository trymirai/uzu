use std::{fmt::Debug, sync::Arc};

use tokio::runtime::Handle as TokioHandle;

use crate::{
    DownloadError,
    traits::{ActiveDownloadGeneration, BackendEventSender, DownloadBackend, DownloadConfig},
};

pub enum InitialTaskAttachment<B: Backend> {
    None,
    Downloading {
        active_task: B::ActiveTask,
        initial_downloaded_bytes: u64,
        total_bytes: Option<u64>,
    },
}

#[async_trait::async_trait]
pub trait Backend: DownloadBackend + Debug + Clone + Send + Sync + Sized + 'static {
    const RESUME_ARTIFACT_EXTENSION: &'static str;

    fn manager_suffix() -> &'static str;

    fn create_context(tokio_handle: TokioHandle) -> Result<Self::Context, DownloadError>;

    async fn initial_task_attachment(
        _context: &Self::Context,
        _config: Arc<DownloadConfig>,
        _generation: ActiveDownloadGeneration,
        _backend_event_sender: BackendEventSender,
    ) -> Result<InitialTaskAttachment<Self>, DownloadError> {
        Ok(InitialTaskAttachment::None)
    }
}
