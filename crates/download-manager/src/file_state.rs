#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileState {
    Exists,
    Missing,
}

pub type DownloadedFileState = FileState;
pub type CRCFileState = FileState;
pub type ResumeDataFileState = FileState;
