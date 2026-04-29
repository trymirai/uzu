use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum InitialLifecycleState {
    NotDownloaded,
    Paused {
        part_path: PathBuf,
    },
    Downloaded {
        file_path: PathBuf,
        crc_path: Option<PathBuf>,
    },
}
