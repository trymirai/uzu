use objc2_foundation::NSURLSessionDownloadTask;
use serde::{Deserialize, Serialize};

use crate::{Crc32c, DownloadId, file_download::DownloadConfig};

#[derive(Serialize, Deserialize, Debug)]
pub struct AppleTaskDescription {
    #[serde(default)]
    pub download_id: DownloadId,
    pub source_url: String,
    #[serde(default)]
    pub crc32c: Option<Crc32c>,
}

impl From<&DownloadConfig> for AppleTaskDescription {
    fn from(config: &DownloadConfig) -> Self {
        Self {
            download_id: config.download_id,
            source_url: config.source_url.clone(),
            crc32c: config.expected_crc32c.clone(),
        }
    }
}

impl AppleTaskDescription {
    pub fn of(task: &NSURLSessionDownloadTask) -> Option<Self> {
        serde_json::from_str(&task.taskDescription()?.to_string()).ok()
    }
}
