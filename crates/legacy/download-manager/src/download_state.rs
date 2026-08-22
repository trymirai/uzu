use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[deprecated(note = "use FileDownloadState or FileDownloadGroupState")]
pub enum DownloadState {
    Downloading,
    Paused,
    NotDownloaded,
    Interrupted,
    Completed,
    Error,
}
