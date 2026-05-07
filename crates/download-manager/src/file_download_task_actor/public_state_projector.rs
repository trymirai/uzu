use crate::{
    FileDownloadState,
    file_download_task_actor::{DownloadActorState, ProgressCounters, PublicProjection},
    traits::{DownloadBackend, DownloadConfig},
};

pub fn project_runtime_public_state<B: DownloadBackend>(
    lifecycle_state: &DownloadActorState<B>,
    projection: &PublicProjection,
    progress_counters: ProgressCounters,
    config: &DownloadConfig,
) -> FileDownloadState {
    match projection {
        PublicProjection::StickyError(message) => FileDownloadState::error(message.clone()),
        PublicProjection::LockedByOther(manager_id) => FileDownloadState::locked_by_other(manager_id.clone()),
        PublicProjection::None => match lifecycle_state {
            DownloadActorState::NotDownloaded => FileDownloadState::not_downloaded(config.expected_bytes.unwrap_or(0)),
            DownloadActorState::Paused {
                ..
            } => FileDownloadState::paused(
                progress_counters.downloaded_bytes,
                fallback_total_bytes(progress_counters, config.expected_bytes),
            ),
            DownloadActorState::Downloaded => {
                let total_bytes = config.expected_bytes.unwrap_or(progress_counters.total_bytes);
                FileDownloadState::downloaded(total_bytes)
            },
            DownloadActorState::Downloading {
                ..
            } => FileDownloadState::downloading(
                progress_counters.downloaded_bytes,
                fallback_total_bytes(progress_counters, config.expected_bytes),
            ),
        },
    }
}

fn fallback_total_bytes(
    progress_counters: ProgressCounters,
    expected_bytes: Option<u64>,
) -> u64 {
    if progress_counters.total_bytes == 0 {
        expected_bytes.unwrap_or(0)
    } else {
        progress_counters.total_bytes
    }
}
