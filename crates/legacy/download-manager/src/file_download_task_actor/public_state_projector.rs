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
        PublicProjection::StickyError(error) => FileDownloadState::error_with_progress(
            progress_counters.downloaded_bytes,
            fallback_total_bytes(progress_counters, config.expected_bytes),
            error.to_string(),
        ),
        PublicProjection::LockedByOther(manager_id) => FileDownloadState::locked_by_other_with_progress(
            progress_counters.downloaded_bytes,
            fallback_total_bytes(progress_counters, config.expected_bytes),
            manager_id.clone(),
        ),
        PublicProjection::None => match lifecycle_state {
            DownloadActorState::NotDownloaded => {
                FileDownloadState::not_downloaded(fallback_total_bytes(progress_counters, config.expected_bytes))
            },
            DownloadActorState::Paused {
                ..
            } => FileDownloadState::paused(
                progress_counters.downloaded_bytes,
                fallback_total_bytes(progress_counters, config.expected_bytes),
            ),
            DownloadActorState::Downloaded => {
                let total_bytes = config
                    .expected_bytes
                    .or(progress_counters.total_bytes)
                    .unwrap_or(progress_counters.downloaded_bytes);
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
    progress_counters.total_bytes.or(expected_bytes).unwrap_or(0)
}
