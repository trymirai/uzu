use uzu::storage::types::DownloadPhase;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoragePhaseKind {
    NotDownloaded,
    Downloading,
    Paused,
    Downloaded,
    Error,
}

impl StoragePhaseKind {
    pub(super) fn matches(
        self,
        phase: &DownloadPhase,
    ) -> bool {
        match self {
            Self::NotDownloaded => matches!(phase, DownloadPhase::NotDownloaded {}),
            Self::Downloading => matches!(phase, DownloadPhase::Downloading {}),
            Self::Paused => matches!(phase, DownloadPhase::Paused {}),
            Self::Downloaded => matches!(phase, DownloadPhase::Downloaded {}),
            Self::Error => matches!(phase, DownloadPhase::Error { .. }),
        }
    }
}
