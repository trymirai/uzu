use uzu::storage::types::DownloadPhase;

#[derive(Clone)]
pub(super) struct ModelViewModel {
    pub id: String,
    pub name: String,
    pub size: String,
    pub bytes: i64,
    pub quant: String,
    pub phase: DownloadPhase,
    pub progress: f32,
    pub is_mirai: bool,
    pub recommended: bool,
}

impl ModelViewModel {
    pub(super) fn installed(&self) -> bool {
        matches!(self.phase, DownloadPhase::Downloaded {})
    }
    pub(super) fn downloading(&self) -> bool {
        matches!(self.phase, DownloadPhase::Downloading {})
    }
    pub(super) fn paused(&self) -> bool {
        matches!(self.phase, DownloadPhase::Paused {})
    }
}
