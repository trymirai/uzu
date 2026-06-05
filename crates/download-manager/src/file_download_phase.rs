use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FileDownloadPhase {
    NotDownloaded,
    Downloading,
    Paused,
    Downloaded,
    LockedByOther(String),
    Error(String),
}
