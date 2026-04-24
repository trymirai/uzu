use download_manager::{FileDownloadPhase, FileDownloadState};
use serde::{Deserialize, Serialize};

use crate::storage::types::DownloadPhase;

#[bindings::export(ClassCloneable)]
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
    #[bindings::export(Getter)]
    pub fn progress(&self) -> f32 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.downloaded_bytes as f32 / self.total_bytes as f32
        }
    }

    #[bindings::export(Getter)]
    pub fn is_in_progress(&self) -> bool {
        self.phase.is_in_progress()
    }

    #[bindings::export(Getter)]
    pub fn can_pause(&self) -> bool {
        self.phase.can_pause()
    }

    #[bindings::export(Getter)]
    pub fn can_delete(&self) -> bool {
        self.phase.can_delete()
    }
}

pub fn reduce_file_download_states(file_states: &[FileDownloadState]) -> DownloadState {
    if file_states.is_empty() {
        return DownloadState::not_downloaded(0);
    }

    let total_bytes: u64 = file_states.iter().map(|f| f.total_bytes).sum();
    let downloaded_bytes: u64 = file_states.iter().map(|f| f.downloaded_bytes).sum();

    let all_downloaded = file_states.iter().all(|f| matches!(f.phase, FileDownloadPhase::Downloaded));
    let any_downloaded = file_states.iter().any(|f| matches!(f.phase, FileDownloadPhase::Downloaded));
    let any_downloading = file_states.iter().any(|f| matches!(f.phase, FileDownloadPhase::Downloading));
    let any_paused = file_states.iter().any(|f| matches!(f.phase, FileDownloadPhase::Paused));
    let any_error = file_states.iter().any(|f| matches!(f.phase, FileDownloadPhase::Error(_)));
    let any_locked = file_states.iter().any(|f| matches!(f.phase, FileDownloadPhase::LockedByOther(_)));

    if all_downloaded {
        return DownloadState::downloaded(total_bytes as i64);
    }

    // Locked takes precedence over other in-progress states
    if any_locked {
        return DownloadState::locked(downloaded_bytes as i64, total_bytes as i64);
    }

    if any_downloading {
        return DownloadState::downloading(downloaded_bytes as i64, total_bytes as i64);
    }

    if any_paused {
        return DownloadState::paused(downloaded_bytes as i64, total_bytes as i64);
    }

    if any_error {
        if let Some(error_state) = file_states.iter().find(|f| matches!(f.phase, FileDownloadPhase::Error(_))) {
            if let FileDownloadPhase::Error(err) = &error_state.phase {
                return DownloadState::error(err.clone());
            }
        }
    }

    // If some files are downloaded but not all, and nothing is actively
    // downloading, treat as paused (partial progress, can resume)
    if any_downloaded && downloaded_bytes > 0 {
        return DownloadState::paused(downloaded_bytes as i64, total_bytes as i64);
    }

    DownloadState::not_downloaded(total_bytes as i64)
}
