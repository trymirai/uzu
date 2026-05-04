use std::path::{Path, PathBuf};

use crate::traits::{CancelOutcome, DownloadBackend};

#[async_trait::async_trait]
pub trait ActiveTask: Send + Sync + Sized {
    type Backend: DownloadBackend<ActiveTask = Self>;

    async fn pause(
        self,
        destination: &Path,
    ) -> Result<PathBuf, <Self::Backend as DownloadBackend>::Error>;

    async fn cancel(
        self,
        destination: &Path,
    ) -> CancelOutcome;
}
