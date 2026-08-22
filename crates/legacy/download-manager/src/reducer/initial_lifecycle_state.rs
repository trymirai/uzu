use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InitialLifecycleState {
    NotDownloaded,
    Paused {
        part_path: PathBuf,
    },
    Downloaded,
}

impl InitialLifecycleState {
    pub fn name(&self) -> &'static str {
        match self {
            Self::NotDownloaded => "NotDownloaded",
            Self::Paused {
                ..
            } => "Paused",
            Self::Downloaded => "Downloaded",
        }
    }
}
