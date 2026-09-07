use serde::{Deserialize, Serialize};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DownloadPhase {
    NotDownloaded {},
    Downloading {},
    Paused {},
    Downloaded {},
    LockedByOther {
        manager_id: String,
    },
    Error {
        message: String,
    },
}

impl DownloadPhase {
    pub fn is_in_progress(&self) -> bool {
        matches!(self, Self::Downloading {})
    }

    pub fn can_pause(&self) -> bool {
        matches!(self, Self::Downloading {})
    }

    pub fn can_delete(&self) -> bool {
        !matches!(self, Self::NotDownloaded {})
    }
}
