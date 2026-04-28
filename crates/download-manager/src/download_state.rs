#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadState {
    Downloading,
    Paused,
    NotDownloaded,
    Interrupted,
    Completed,
    Error,
}
