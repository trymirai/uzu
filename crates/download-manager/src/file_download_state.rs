use crate::FileDownloadPhase;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileDownloadState {
    /// Total size of the file in bytes.
    pub total_bytes: u64,
    /// Bytes already downloaded (meaningful in Downloading / Paused).
    pub downloaded_bytes: u64,
    /// Current phase of the download.
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

    pub fn error(err: String) -> Self {
        Self {
            total_bytes: 0,
            downloaded_bytes: 0,
            phase: FileDownloadPhase::Error(err),
        }
    }
}
