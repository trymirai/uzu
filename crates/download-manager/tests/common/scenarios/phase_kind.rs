use download_manager::FileDownloadPhase;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhaseKind {
    Downloaded,
    Error,
}

impl PhaseKind {
    pub fn matches(
        self,
        phase: &FileDownloadPhase,
    ) -> bool {
        match self {
            Self::Downloaded => matches!(phase, FileDownloadPhase::Downloaded),
            Self::Error => matches!(phase, FileDownloadPhase::Error(_)),
        }
    }
}
