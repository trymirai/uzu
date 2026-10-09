use serde::{Deserialize, Serialize};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DownloadPhase {
    NotDownloaded {},
    Downloading {},
    Paused {},
    Downloaded {},
    Locked {
        manager_id: String,
    },
    Error {
        message: String,
    },
    Initializing {},
}

impl DownloadPhase {
    pub fn is_in_progress(&self) -> bool {
        matches!(self, Self::Initializing {} | Self::Downloading {} | Self::Locked { .. })
    }
}
