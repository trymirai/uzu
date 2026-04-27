use download_manager::FileDownloadPhase;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhaseKind {
    NotDownloaded,
    Paused,
    Downloaded,
    Error,
}

impl PhaseKind {
    pub fn matches(
        self,
        phase: &FileDownloadPhase,
    ) -> bool {
        match self {
            Self::NotDownloaded => matches!(phase, FileDownloadPhase::NotDownloaded),
            Self::Paused => matches!(phase, FileDownloadPhase::Paused),
            Self::Downloaded => matches!(phase, FileDownloadPhase::Downloaded),
            Self::Error => matches!(phase, FileDownloadPhase::Error(_)),
        }
    }
}
