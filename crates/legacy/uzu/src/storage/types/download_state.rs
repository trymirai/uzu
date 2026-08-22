use serde::{Deserialize, Serialize};

use crate::storage::types::DownloadPhase;

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DownloadState {
    pub total_bytes: i64,
    pub downloaded_bytes: i64,
    pub phase: DownloadPhase,
}

impl DownloadState {
    pub fn not_downloaded(total_bytes: i64) -> Self {
        Self {
            total_bytes,
            downloaded_bytes: 0,
            phase: DownloadPhase::NotDownloaded {},
        }
    }

    pub fn downloading(
        downloaded_bytes: i64,
        total_bytes: i64,
    ) -> Self {
        Self {
            total_bytes,
            downloaded_bytes,
            phase: DownloadPhase::Downloading {},
        }
    }

    pub fn paused(
        downloaded_bytes: i64,
        total_bytes: i64,
    ) -> Self {
        Self {
            total_bytes,
            downloaded_bytes,
            phase: DownloadPhase::Paused {},
        }
    }

    pub fn downloaded(total_bytes: i64) -> Self {
        Self {
            total_bytes,
            downloaded_bytes: total_bytes,
            phase: DownloadPhase::Downloaded {},
        }
    }

    pub fn locked(
        downloaded_bytes: i64,
        total_bytes: i64,
    ) -> Self {
        Self {
            total_bytes,
            downloaded_bytes,
            phase: DownloadPhase::Locked {},
        }
    }

    pub fn error(error_message: String) -> Self {
        Self {
            total_bytes: 0,
            downloaded_bytes: 0,
            phase: DownloadPhase::Error {
                message: error_message,
            },
        }
    }
}

#[bindings::export(Implementation)]
impl DownloadState {
    #[bindings::export(Method(Getter))]
    pub fn progress(&self) -> f32 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.downloaded_bytes as f32 / self.total_bytes as f32
        }
    }

    #[bindings::export(Method(Getter))]
    pub fn is_in_progress(&self) -> bool {
        self.phase.is_in_progress()
    }

    #[bindings::export(Method(Getter))]
    pub fn can_pause(&self) -> bool {
        self.phase.can_pause()
    }

    #[bindings::export(Method(Getter))]
    pub fn can_delete(&self) -> bool {
        self.phase.can_delete()
    }

    #[bindings::export(Method(Getter))]
    pub fn name(&self) -> String {
        match &self.phase {
            DownloadPhase::NotDownloaded {} => "Not Downloaded".to_string(),
            DownloadPhase::Downloading {} => "Downloading".to_string(),
            DownloadPhase::Paused {} => "Paused".to_string(),
            DownloadPhase::Downloaded {} => "Downloaded".to_string(),
            DownloadPhase::Locked {} => "Locked".to_string(),
            DownloadPhase::Error {
                ..
            } => "Error".to_string(),
        }
    }
}
