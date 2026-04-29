use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FileDownloadEvent {
    DownloadCompleted {
        tmp_path: PathBuf,
        final_destination: PathBuf,
    },
    ProgressUpdate {
        bytes_written: u64,
        total_bytes_written: u64,
        total_bytes_expected: u64,
    },
    Error {
        message: String,
    },
}
