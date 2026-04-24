use uzu::storage::types::DownloadPhase;

use crate::app::ModelWithState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Section {
    Installed,
    Downloading,
    Available,
}

impl Section {
    pub fn next(&self) -> Self {
        match self {
            Section::Installed => Section::Downloading,
            Section::Downloading => Section::Available,
            Section::Available => Section::Installed,
        }
    }

    pub fn prev(&self) -> Self {
        match self {
            Section::Installed => Section::Available,
            Section::Downloading => Section::Installed,
            Section::Available => Section::Downloading,
        }
    }

    pub fn title(&self) -> &'static str {
        match self {
            Section::Installed => "Installed",
            Section::Downloading => "Downloading",
            Section::Available => "Available",
        }
    }

    /// Determine which section a model belongs to based on its state
    pub fn for_model(model_with_state: &ModelWithState) -> Self {
        match model_with_state.state.phase {
            DownloadPhase::Downloaded {} => Section::Installed,
            DownloadPhase::Downloading {}
            | DownloadPhase::Paused {}
            | DownloadPhase::Locked {}
            | DownloadPhase::Error {
                ..
            } => Section::Downloading,
            DownloadPhase::NotDownloaded {} => Section::Available,
        }
    }

    pub fn all() -> [Section; 3] {
        [Section::Installed, Section::Downloading, Section::Available]
    }
}
