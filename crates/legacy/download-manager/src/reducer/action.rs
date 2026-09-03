use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::FileCheck;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    DeleteFile {
        path: PathBuf,
    },
    DeleteIntegrityCache {
        path: PathBuf,
    },
    DeleteResumeArtifact {
        path: PathBuf,
    },
    SaveIntegrityCache {
        destination: PathBuf,
        file_check: FileCheck,
    },
}
