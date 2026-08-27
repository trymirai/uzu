use serde::{Deserialize, Serialize};
use serde_json::{Error as JsonError, from_str, to_string};

use crate::FileCheck;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DownloadInfo {
    pub source_url: String,
    pub destination_path: String,
    #[serde(default)]
    pub crc32c: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_check: Option<FileCheck>,
}

impl DownloadInfo {
    pub fn new<U: Into<String>, P: Into<String>>(
        source_url: U,
        destination_path: P,
        file_check: FileCheck,
    ) -> Self {
        let crc32c = file_check.expected_crc();
        Self {
            source_url: source_url.into(),
            destination_path: destination_path.into(),
            crc32c,
            file_check: Some(file_check),
        }
    }

    pub fn resolved_file_check(&self) -> FileCheck {
        self.file_check.clone().or_else(|| self.crc32c.clone().map(FileCheck::CRC)).unwrap_or(FileCheck::None)
    }

    pub fn to_json(&self) -> Result<String, JsonError> {
        to_string(self)
    }

    pub fn from_json(json: &str) -> Result<Self, JsonError> {
        from_str(json)
    }
}
