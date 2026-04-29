use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FileState {
    Exists,
    Missing,
}

pub type DownloadedFileState = FileState;
pub type CRCFileState = FileState;
pub type ResumeDataFileState = FileState;
