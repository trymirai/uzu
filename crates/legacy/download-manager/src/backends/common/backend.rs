use std::{fmt::Debug, sync::Arc};

use kiban::rt::RuntimeHandle;

use crate::{
    DownloadError,
    backends::common::{ActiveDownloadGeneration, BackendEventSender, DownloadConfig, InitialTaskAttachment},
    lock_manager::DestinationLockLease,
    traits::DownloadBackend,
};

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
pub trait Backend: DownloadBackend + Debug + Clone + Send + Sync + Sized + 'static {
    const RESUME_ARTIFACT_EXTENSION: &'static str;
    const SUPPORTS_INITIAL_TASK_ATTACHMENT: bool = false;

    fn manager_suffix() -> &'static str;

    fn create_context(runtime_handle: RuntimeHandle) -> Result<Self::Context, DownloadError>;

    async fn initial_task_attachment(
        _context: &Self::Context,
        _config: Arc<DownloadConfig>,
        _generation: ActiveDownloadGeneration,
        _backend_event_sender: BackendEventSender,
        _destination_lease: &DestinationLockLease,
    ) -> Result<InitialTaskAttachment<Self>, DownloadError> {
        Ok(InitialTaskAttachment::None)
    }
}
