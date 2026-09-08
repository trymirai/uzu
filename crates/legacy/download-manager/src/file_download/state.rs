use crate::{
    DownloadPhase, DownloadState, backends::ActiveTask, file_download::DownloadConfig, locks::DestinationLock,
};

pub enum State {
    NotDownloaded,
    Paused {
        downloaded_bytes: u64,
    },
    Downloading {
        active_task: Box<dyn ActiveTask>,
        lock: DestinationLock,
        downloaded_bytes: u64,
        total_bytes: Option<u64>,
    },
    Downloaded {
        total_bytes: u64,
    },
    Locked {
        manager_id: String,
        downloaded_bytes: u64,
    },
    Failed {
        message: String,
    },
}

impl State {
    pub fn download_state(
        &self,
        config: &DownloadConfig,
    ) -> DownloadState {
        let (downloaded_bytes, observed_total, phase) = match self {
            Self::NotDownloaded => (0, None, DownloadPhase::NotDownloaded {}),
            Self::Paused {
                downloaded_bytes,
            } => (*downloaded_bytes, None, DownloadPhase::Paused {}),
            Self::Downloading {
                downloaded_bytes,
                total_bytes,
                ..
            } => (*downloaded_bytes, *total_bytes, DownloadPhase::Downloading {}),
            Self::Downloaded {
                total_bytes,
            } => (*total_bytes, Some(*total_bytes), DownloadPhase::Downloaded {}),
            Self::Locked {
                manager_id,
                downloaded_bytes,
            } => (
                *downloaded_bytes,
                None,
                DownloadPhase::Locked {
                    manager_id: manager_id.clone(),
                },
            ),
            Self::Failed {
                message,
            } => (
                0,
                None,
                DownloadPhase::Error {
                    message: message.clone(),
                },
            ),
        };
        DownloadState {
            total_bytes: config.expected_bytes.or(observed_total).unwrap_or(downloaded_bytes) as i64,
            downloaded_bytes: downloaded_bytes as i64,
            phase,
        }
    }

    pub fn is_downloading(&self) -> bool {
        matches!(self, Self::Downloading { .. })
    }
}
