use std::path::PathBuf;

use crate::{BearerToken, Checksum, DownloadId, locks::LockOwner};

pub struct DownloadConfig {
    pub download_id: DownloadId,
    pub source_url: String,
    pub bearer_token: Option<BearerToken>,
    pub destination: PathBuf,
    pub resume_artifact_path: PathBuf,
    pub expected_checksum: Option<Checksum>,
    pub expected_bytes: Option<u64>,
    pub owner: LockOwner,
}
