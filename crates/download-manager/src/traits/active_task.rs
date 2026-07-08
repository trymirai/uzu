use std::path::{Path, PathBuf};

use crate::traits::{CancelOutcome, DownloadBackend};

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
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
