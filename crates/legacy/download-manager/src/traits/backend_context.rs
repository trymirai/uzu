use std::{path::Path, sync::Arc};

use crate::{
    backends::common::{ActiveDownloadGeneration, BackendEventSender, DownloadConfig},
    lock_manager::DestinationLockLease,
    traits::DownloadBackend,
};

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
pub trait BackendContext: Send + Sync + Sized {
    type Backend: DownloadBackend<Context = Self>;

    async fn download(
        &self,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        backend_event_sender: BackendEventSender,
        destination_lease: &DestinationLockLease,
    ) -> Result<<Self::Backend as DownloadBackend>::ActiveTask, <Self::Backend as DownloadBackend>::Error>;

    async fn resume(
        &self,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        resume_artifact_path: &Path,
        backend_event_sender: BackendEventSender,
        destination_lease: &DestinationLockLease,
    ) -> Result<<Self::Backend as DownloadBackend>::ActiveTask, <Self::Backend as DownloadBackend>::Error>;
}
