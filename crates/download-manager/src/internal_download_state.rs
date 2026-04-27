use crate::{DownloadError, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateTransitionAction {
    Download,
    Pause,
    Cancel,
}

#[derive(Debug)]
pub enum InternalDownloadState<Task> {
    Downloaded,
    Downloading {
        task: Task,
    },
    Paused {
        part_path: PathBuf,
    },
    NotDownloaded,
}

impl<Task> InternalDownloadState<Task> {
    pub fn can_transition(
        &self,
        action: StateTransitionAction,
    ) -> Result<(), DownloadError> {
        match (self, action) {
            (InternalDownloadState::NotDownloaded, StateTransitionAction::Download) => Ok(()),
            (
                InternalDownloadState::Paused {
                    ..
                },
                StateTransitionAction::Download,
            ) => Ok(()),
            (
                InternalDownloadState::Downloading {
                    ..
                },
                StateTransitionAction::Download,
            ) => Ok(()),
            (InternalDownloadState::Downloaded, StateTransitionAction::Download) => Ok(()),

            (
                InternalDownloadState::Downloading {
                    ..
                },
                StateTransitionAction::Pause,
            ) => Ok(()),
            (
                InternalDownloadState::Paused {
                    ..
                },
                StateTransitionAction::Pause,
            ) => Ok(()),
            (InternalDownloadState::Downloaded, StateTransitionAction::Pause) => Ok(()),
            (InternalDownloadState::NotDownloaded, StateTransitionAction::Pause) => {
                Err(DownloadError::InvalidStateTransition)
            },

            (_, StateTransitionAction::Cancel) => Ok(()),
        }
    }
}
