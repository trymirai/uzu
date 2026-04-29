use crate::{
    FileDownloadState,
    file_download_task_actor::{LifecycleState, ProgressCounters, PublicProjection},
    reducer::InitialLifecycleState,
    traits::{DownloadBackend, DownloadConfig},
};

pub fn project_public_state(
    lifecycle_state: &InitialLifecycleState,
    projection: &PublicProjection,
    progress_counters: ProgressCounters,
    config: &DownloadConfig,
) -> FileDownloadState {
    match projection {
        PublicProjection::StickyError(message) => FileDownloadState::error(message.clone()),
        PublicProjection::LockedByOther(manager_id) => FileDownloadState::locked_by_other(manager_id.clone()),
        PublicProjection::None => match lifecycle_state {
            InitialLifecycleState::NotDownloaded => {
                FileDownloadState::not_downloaded(config.expected_bytes.unwrap_or(0))
            },
            InitialLifecycleState::Paused {
                ..
            } => FileDownloadState::paused(
                progress_counters.downloaded_bytes,
                fallback_total_bytes(progress_counters, config.expected_bytes),
            ),
            InitialLifecycleState::Downloaded {
                ..
            } => {
                let total_bytes = config.expected_bytes.unwrap_or(progress_counters.total_bytes);
                FileDownloadState::downloaded(total_bytes)
            },
        },
    }
}

pub fn project_runtime_public_state<B: DownloadBackend>(
    lifecycle_state: &LifecycleState<B>,
    projection: &PublicProjection,
    progress_counters: ProgressCounters,
    config: &DownloadConfig,
) -> FileDownloadState {
    match projection {
        PublicProjection::StickyError(message) => FileDownloadState::error(message.clone()),
        PublicProjection::LockedByOther(manager_id) => FileDownloadState::locked_by_other(manager_id.clone()),
        PublicProjection::None => match lifecycle_state {
            LifecycleState::NotDownloaded => FileDownloadState::not_downloaded(config.expected_bytes.unwrap_or(0)),
            LifecycleState::Paused {
                ..
            } => FileDownloadState::paused(
                progress_counters.downloaded_bytes,
                fallback_total_bytes(progress_counters, config.expected_bytes),
            ),
            LifecycleState::Downloaded {
                ..
            } => {
                let total_bytes = config.expected_bytes.unwrap_or(progress_counters.total_bytes);
                FileDownloadState::downloaded(total_bytes)
            },
            LifecycleState::Downloading {
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
