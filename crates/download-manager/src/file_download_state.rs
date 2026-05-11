use serde::{Deserialize, Serialize};

use crate::FileDownloadPhase;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FileDownloadState {
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub phase: FileDownloadPhase,
}

impl FileDownloadState {
    pub fn not_downloaded(total_bytes: u64) -> Self {
        Self {
            total_bytes,
            downloaded_bytes: 0,
            phase: FileDownloadPhase::NotDownloaded,
        }
    }

    pub fn downloading(
        downloaded_bytes: u64,
        total_bytes: u64,
    ) -> Self {
        Self {
            total_bytes,
            downloaded_bytes,
            phase: FileDownloadPhase::Downloading,
        }
    }

    pub fn paused(
        downloaded_bytes: u64,
        total_bytes: u64,
    ) -> Self {
        Self {
            total_bytes,
            downloaded_bytes,
            phase: FileDownloadPhase::Paused,
        }
    }

    pub fn downloaded(total_bytes: u64) -> Self {
        Self {
            total_bytes,
            downloaded_bytes: total_bytes,
            phase: FileDownloadPhase::Downloaded,
        }
    }

    pub fn locked_by_other(manager_id: String) -> Self {
        Self {
            total_bytes: 0,
            downloaded_bytes: 0,
            phase: FileDownloadPhase::LockedByOther(manager_id),
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            total_bytes: 0,
            downloaded_bytes: 0,
            phase: FileDownloadPhase::Error(message),
        }
    }
}
