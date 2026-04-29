use std::path::PathBuf;

use crate::traits::{ActiveDownloadGeneration, DownloadBackend};

#[derive(Debug)]
pub enum LifecycleState<B: DownloadBackend> {
    NotDownloaded,
    Paused {
        part_path: PathBuf,
    },
    Downloaded {
        file_path: PathBuf,
        crc_path: Option<PathBuf>,
    },
    Downloading {
        active_task: Option<B::ActiveTask>,
        generation: ActiveDownloadGeneration,
    },
}

impl<B: DownloadBackend> LifecycleState<B> {
    pub fn is_downloading_generation(
        &self,
        event_generation: ActiveDownloadGeneration,
    ) -> bool {
        matches!(
            self,
            Self::Downloading {
                generation,
                ..
            } if *generation == event_generation
        )
    }
}
