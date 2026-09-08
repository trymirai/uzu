use std::path::PathBuf;

use uuid::Uuid;

use crate::{Crc32c, DownloadId};

pub struct DownloadConfig {
    pub download_id: DownloadId,
    pub source_url: String,
    pub destination: PathBuf,
    pub resume_artifact_path: PathBuf,
    pub expected_crc32c: Option<Crc32c>,
    pub expected_bytes: Option<u64>,
    pub manager_id: String,
    pub manager_instance_id: Uuid,
}
