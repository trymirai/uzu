use crate::{
    DownloadPhase, DownloadState,
    backends::common::DownloadConfig,
    file_download_task_actor::{DownloadActorState, ProgressCounters, PublicProjection},
    traits::DownloadBackend,
};

pub fn project_runtime_public_state<B: DownloadBackend>(
    lifecycle_state: &DownloadActorState<B>,
    projection: &PublicProjection,
    progress_counters: ProgressCounters,
    config: &DownloadConfig,
) -> DownloadState {
    let total_bytes = config.expected_bytes.unwrap_or(progress_counters.total_bytes) as i64;
    let (downloaded_bytes, phase) = match projection {
        PublicProjection::StickyError(message) => (
            0,
            DownloadPhase::Error {
                message: message.clone(),
            },
        ),
        PublicProjection::LockedByOther(manager_id) => (
            0,
            DownloadPhase::LockedByOther {
                manager_id: manager_id.clone(),
            },
        ),
        PublicProjection::None => match lifecycle_state {
            DownloadActorState::NotDownloaded => (0, DownloadPhase::NotDownloaded {}),
            DownloadActorState::Paused {
                ..
            } => (progress_counters.downloaded_bytes as i64, DownloadPhase::Paused {}),
            DownloadActorState::Downloading {
                ..
            } => (progress_counters.downloaded_bytes as i64, DownloadPhase::Downloading {}),
            DownloadActorState::Downloaded => (total_bytes, DownloadPhase::Downloaded {}),
        },
    };
    DownloadState {
        total_bytes,
        downloaded_bytes,
        phase,
    }
}
