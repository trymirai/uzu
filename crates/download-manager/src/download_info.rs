use serde::{Deserialize, Serialize};

/// DownloadInfo is a small, serializable struct that describes a download.
/// It is designed to be encoded/decoded to JSON and stored in NSURLSessionTask.taskDescription.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadInfo {
    /// The original source URL for the download.
    pub source_url: String,
    /// The absolute or relative destination path where the file should be persisted.
    pub destination_path: String,
    /// Optional CRC32C checksum in base64 format for file validation.
    #[serde(default)]
    pub crc32c: Option<String>,
}

impl DownloadInfo {
    /// Construct a new DownloadInfo.
    pub fn new<U: Into<String>, P: Into<String>>(
        source_url: U,
        destination_path: P,
    ) -> Self {
        Self {
            source_url: source_url.into(),
            destination_path: destination_path.into(),
            crc32c: None,
        }
    }

    pub fn with_crc<U: Into<String>, P: Into<String>, C: Into<String>>(
        source_url: U,
        destination_path: P,
        crc32c: C,
    ) -> Self {
        Self {
            source_url: source_url.into(),
            destination_path: destination_path.into(),
            crc32c: Some(crc32c.into()),
        }
    }

    /// Serialize this DownloadInfo to a compact JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize a DownloadInfo from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}
