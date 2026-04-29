use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{DownloadId, FileCheck};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DownloadConfig {
    pub download_id: DownloadId,
    pub source_url: String,
    pub destination: PathBuf,
    pub file_check: FileCheck,
    pub expected_bytes: Option<u64>,
    pub manager_id: String,
}
