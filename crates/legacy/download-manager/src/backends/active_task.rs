use std::path::Path;

use crate::DownloadError;

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
pub trait ActiveTask: Send + Sync {
    async fn pause(
        self: Box<Self>,
        resume_artifact_path: &Path,
    ) -> Result<(), DownloadError>;

    async fn cancel(self: Box<Self>);
}
