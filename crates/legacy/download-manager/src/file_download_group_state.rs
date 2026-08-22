use std::{path::PathBuf, sync::Arc};

use crate::{DownloadError, RelativeFilePath};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FileDownloadGroupPhase {
    NotDownloaded,
    Downloading,
    Paused,
    Downloaded,
    Locked,
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileDownloadFailure {
    pub relative_path: RelativeFilePath,
    pub error: DownloadError,
}

impl FileDownloadFailure {
    pub fn new(
        relative_path: RelativeFilePath,
        error: DownloadError,
    ) -> Self {
        Self {
            relative_path,
            error,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileDownloadGroupState {
    pub phase: FileDownloadGroupPhase,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub completed_files: usize,
    pub total_files: usize,
    pub failures: Arc<[FileDownloadFailure]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FileDownloadGroupOperation {
    Create,
    Pause,
    Cancel,
}

impl std::fmt::Display for FileDownloadGroupOperation {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let name = match self {
            Self::Create => "create",
            Self::Pause => "pause",
            Self::Cancel => "cancel",
        };
        formatter.write_str(name)
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum FileDownloadGroupError {
    #[error("file download group actor stopped")]
    ActorStopped,
    #[error("another live file download group owns destination root: {destination_root}")]
    RootConflict {
        destination_root: PathBuf,
    },
    #[error("file download group {operation} failed for {} file(s)", .failures.len())]
    FileFailures {
        operation: FileDownloadGroupOperation,
        failures: Arc<[FileDownloadFailure]>,
    },
}

impl FileDownloadGroupError {
    pub(crate) fn file_failures(
        operation: FileDownloadGroupOperation,
        failures: Vec<FileDownloadFailure>,
    ) -> Self {
        debug_assert!(!failures.is_empty());
        Self::FileFailures {
            operation,
            failures: failures.into(),
        }
    }

    pub fn failures(&self) -> &[FileDownloadFailure] {
        match self {
            Self::ActorStopped
            | Self::RootConflict {
                ..
            } => &[],
            Self::FileFailures {
                failures,
                ..
            } => failures,
        }
    }
}
