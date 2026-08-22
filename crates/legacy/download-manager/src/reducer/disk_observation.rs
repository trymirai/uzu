use std::path::PathBuf;

use crate::{FileCheck, FileState};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiskObservation {
    pub destination_state: FileState,
    pub crc_state: FileState,
    pub resume_state: FileState,
    pub destination_size: Option<u64>,
    pub resume_size: Option<u64>,
    pub file_check: FileCheck,
    pub expected_bytes: Option<u64>,
    pub destination_path: PathBuf,
    pub crc_path: Option<PathBuf>,
    pub resume_artifact_path: Option<PathBuf>,
}
