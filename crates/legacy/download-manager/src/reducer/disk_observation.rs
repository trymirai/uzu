use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{FileCheck, FileState};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DiskObservation {
    pub destination_state: FileState,
    pub integrity_state: FileState,
    pub resume_state: FileState,
    pub destination_size: Option<u64>,
    pub resume_size: Option<u64>,
    pub file_check: FileCheck,
    pub expected_bytes: Option<u64>,
    pub destination_path: PathBuf,
    pub integrity_path: PathBuf,
    pub resume_artifact_path: Option<PathBuf>,
}
