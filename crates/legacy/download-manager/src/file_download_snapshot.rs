use crate::{DownloadError, FileDownloadPhase, FileDownloadState};

/// The complete observable state of one file download.
///
/// Keeping the public state and its typed failure in one watched value prevents
/// callers from observing a new phase with an error left over from an older
/// transition (or the reverse).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileDownloadSnapshot {
    pub state: FileDownloadState,
    pub failure: Option<DownloadError>,
    /// The known total size. `None` means the source has not supplied one.
    pub total_bytes: Option<u64>,
}

impl FileDownloadSnapshot {
    pub fn new(
        state: FileDownloadState,
        failure: Option<DownloadError>,
    ) -> Self {
        let total_bytes = (state.total_bytes > 0 || matches!(state.phase, FileDownloadPhase::Downloaded))
            .then_some(state.total_bytes);
        Self {
            state,
            failure,
            total_bytes,
        }
    }

    pub(crate) fn with_total_bytes(
        state: FileDownloadState,
        failure: Option<DownloadError>,
        total_bytes: Option<u64>,
    ) -> Self {
        Self {
            state,
            failure,
            total_bytes,
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/file_download_snapshot_test.rs"]
mod tests;
