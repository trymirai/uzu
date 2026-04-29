use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DownloadState {
    Downloading,
    Paused,
    NotDownloaded,
    Interrupted,
    Completed,
    Error,
}
