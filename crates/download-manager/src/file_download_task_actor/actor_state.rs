use std::{fmt, path::PathBuf};

use crate::{
    lock_manager::DestinationLockLease,
    reducer::InitialLifecycleState,
    traits::{ActiveDownloadGeneration, DownloadBackend},
};

pub(crate) enum DownloadActorState<B: DownloadBackend> {
    NotDownloaded,
    Paused {
        part_path: PathBuf,
    },
    Downloading {
        active_task: B::ActiveTask,
        generation: ActiveDownloadGeneration,
        destination_lease: DestinationLockLease,
    },
    Downloaded {
        file_path: PathBuf,
        crc_path: Option<PathBuf>,
    },
}

impl<B: DownloadBackend> DownloadActorState<B> {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::NotDownloaded => "NotDownloaded",
            Self::Paused {
                ..
            } => "Paused",
            Self::Downloading {
                ..
            } => "Downloading",
            Self::Downloaded {
                ..
            } => "Downloaded",
        }
    }
}

impl<B: DownloadBackend> fmt::Debug for DownloadActorState<B> {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::NotDownloaded => formatter.debug_struct("NotDownloaded").finish(),
            Self::Paused {
                part_path,
            } => formatter.debug_struct("Paused").field("part_path", part_path).finish(),
            Self::Downloading {
                generation,
                ..
            } => formatter.debug_struct("Downloading").field("generation", generation).finish_non_exhaustive(),
            Self::Downloaded {
                file_path,
                crc_path,
            } => {
                formatter.debug_struct("Downloaded").field("file_path", file_path).field("crc_path", crc_path).finish()
            },
        }
    }
}

impl<B: DownloadBackend> From<InitialLifecycleState> for DownloadActorState<B> {
    fn from(initial_lifecycle_state: InitialLifecycleState) -> Self {
        match initial_lifecycle_state {
            InitialLifecycleState::NotDownloaded => Self::NotDownloaded,
            InitialLifecycleState::Paused {
                part_path,
            } => Self::Paused {
                part_path,
            },
            InitialLifecycleState::Downloaded {
                file_path,
                crc_path,
            } => Self::Downloaded {
                file_path,
                crc_path,
            },
        }
    }
}
