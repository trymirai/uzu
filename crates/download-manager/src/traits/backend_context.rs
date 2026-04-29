use std::{path::Path, sync::Arc};

use crate::traits::{ActiveDownloadGeneration, BackendEventSender, DownloadBackend, DownloadConfig};

#[async_trait::async_trait]
pub trait BackendContext: Send + Sync + Sized {
    type Backend: DownloadBackend<Context = Self>;

    async fn download(
        &self,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        backend_event_sender: BackendEventSender,
    ) -> Result<<Self::Backend as DownloadBackend>::ActiveTask, <Self::Backend as DownloadBackend>::Error>;

    async fn resume(
        &self,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        resume_artifact_path: &Path,
        backend_event_sender: BackendEventSender,
    ) -> Result<<Self::Backend as DownloadBackend>::ActiveTask, <Self::Backend as DownloadBackend>::Error>;
}
