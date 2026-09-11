use std::path::PathBuf;

use crate::{Crc32c, DownloadId, locks::LockOwner};

pub struct DownloadConfig {
    pub download_id: DownloadId,
    pub source_url: String,
    pub destination: PathBuf,
    pub resume_artifact_path: PathBuf,
    pub expected_crc32c: Option<Crc32c>,
    pub expected_bytes: Option<u64>,
    pub owner: LockOwner,
}
