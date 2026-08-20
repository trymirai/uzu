use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DownloadInfo {
    pub source_url: String,
    pub destination_path: String,
    #[serde(default)]
    pub crc32c: Option<String>,
}

impl DownloadInfo {
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

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}
