use serde::{Deserialize, Serialize};

use crate::DownloadPhase;

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DownloadState {
    pub total_bytes: i64,
    pub downloaded_bytes: i64,
    pub phase: DownloadPhase,
}

impl Default for DownloadState {
    fn default() -> Self {
        Self {
            total_bytes: 0,
            downloaded_bytes: 0,
            phase: DownloadPhase::NotDownloaded {},
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
            DownloadPhase::LockedByOther {
                ..
            } => "Locked".to_string(),
            DownloadPhase::Error {
                ..
            } => "Error".to_string(),
        }
    }
}
