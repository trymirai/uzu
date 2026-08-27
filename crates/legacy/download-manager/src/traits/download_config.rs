use std::path::PathBuf;

use uuid::Uuid;

use crate::{DownloadId, FileCheck, HttpDownloadRequest};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DownloadConfig {
    pub download_id: DownloadId,
    pub request: HttpDownloadRequest,
    pub destination: PathBuf,
    pub file_check: FileCheck,
    pub expected_bytes: Option<u64>,
    pub manager_id: String,
    pub manager_instance_id: Uuid,
}
