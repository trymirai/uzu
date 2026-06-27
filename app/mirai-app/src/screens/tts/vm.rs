use uzu::storage::types::DownloadPhase;

use crate::models_store::ModelRow;

/// Display data for one TTS model row, projected from a [`ModelRow`].
pub(super) struct TtsVm {
    pub id: String,
    pub name: String,
    pub vendor: String,
    pub icon_url: Option<String>,
    pub size: String,
    pub installed: bool,
    pub downloading: bool,
    pub progress: f32,
}

impl TtsVm {
    pub(super) fn from_row(row: &ModelRow, dark: bool) -> Self {
        Self {
            id: row.id().to_string(),
            name: row.name(),
            vendor: row.vendor().unwrap_or_else(|| "Other".to_string()),
            icon_url: row.icon_url(dark),
            size: crate::screens::local_models::format_size(row.display_size_bytes()),
            installed: row.is_installed(),
            downloading: matches!(
                row.phase(),
                DownloadPhase::Downloading {} | DownloadPhase::Paused {}
            ),
            progress: row.progress(),
        }
    }
}
