use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    DeleteFile {
        path: PathBuf,
    },
    DeleteCrcCache {
        path: PathBuf,
    },
    DeleteResumeArtifact {
        path: PathBuf,
    },
    SaveCrcCache {
        destination: PathBuf,
        crc: String,
    },
}
