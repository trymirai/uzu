use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
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
