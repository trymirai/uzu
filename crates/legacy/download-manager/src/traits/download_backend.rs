use std::{fmt::Debug, future::Future, path::Path};

use kiban::{fs, maybe::MaybeSend};

use crate::traits::{ActiveTask, BackendContext};

pub trait DownloadBackend: Debug + Clone + Send + Sync + 'static {
    type Context: BackendContext<Backend = Self>;
    type ActiveTask: ActiveTask<Backend = Self>;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Retries terminal transport and transient HTTP failures outside the
    /// backend task. Backends with their own retry loop leave this at zero.
    const TERMINAL_RETRY_COUNT: u16 = 0;

    // Default = file size (correct for `.part`-style artifacts). Apple must override:
    // `.resume_data` is a small metadata blob, not the downloaded bytes.
    fn read_resume_progress(part_path: &Path) -> impl Future<Output = Option<u64>> + MaybeSend {
        async move { fs::asyn::file_length(part_path).await.ok() }
    }
}
