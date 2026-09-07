#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadStreamCompletion {
    Completed,
    Paused,
}
