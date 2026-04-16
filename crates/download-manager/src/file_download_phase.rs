#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileDownloadPhase {
    NotDownloaded,
    Downloading,
    Paused,
    Downloaded,
    LockedByOther(String),
    Error(String),
}
