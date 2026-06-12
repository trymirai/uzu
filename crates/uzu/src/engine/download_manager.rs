use download_manager::FileDownloadManagerType;
use serde::{Deserialize, Serialize};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadManagerType {
    #[cfg(target_vendor = "apple")]
    Native,
    Universal,
}

impl Default for DownloadManagerType {
    #[allow(unreachable_code)]
    fn default() -> Self {
        #[cfg(target_vendor = "apple")]
        return Self::Native;
        Self::Universal
    }
}

impl From<DownloadManagerType> for FileDownloadManagerType {
    fn from(value: DownloadManagerType) -> Self {
        match value {
            #[cfg(target_vendor = "apple")]
            DownloadManagerType::Native => Self::Apple,
            DownloadManagerType::Universal => Self::Universal,
        }
    }
}
