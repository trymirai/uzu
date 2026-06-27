use uzu::storage::types::DownloadPhase;

/// One model row within a family detail list.
#[derive(Clone)]
pub(super) struct ModelVm {
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

impl ModelVm {
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

/// A vendor family grouping several [`ModelVm`]s in the family grid.
pub(super) struct FamilyVm {
    pub key: String,
    pub name: String,
    pub vendor: String,
    pub icon_url: Option<String>,
    pub range: Option<String>,
    pub has_mirai: bool,
    pub last_installed_at: u64,
    pub models: Vec<ModelVm>,
}

impl FamilyVm {
    pub(super) fn installed_count(&self) -> usize {
        self.models.iter().filter(|m| m.installed()).count()
    }
}
