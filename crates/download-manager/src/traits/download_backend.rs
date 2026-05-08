use std::{fmt::Debug, path::Path};

use crate::traits::{ActiveTask, BackendContext};

pub trait DownloadBackend: Debug + Clone + Send + Sync + 'static {
    type Context: BackendContext<Backend = Self>;
    type ActiveTask: ActiveTask<Backend = Self>;
    type Error: std::error::Error + Send + Sync + 'static;

    // Default = file size (correct for `.part`-style artifacts). Apple must override:
    // `.resume_data` is a small metadata blob, not the downloaded bytes.
    fn read_resume_progress(part_path: &Path) -> Option<u64> {
        std::fs::metadata(part_path).ok().map(|metadata| metadata.len())
    }
}
