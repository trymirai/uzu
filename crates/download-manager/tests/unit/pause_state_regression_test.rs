use std::path::PathBuf;

use download_manager::{
    FileCheck, FileDownloadPhase, FileDownloadState, InternalDownloadState,
    managers::universal::{AsyncFetcherConfig, FileDownloadTask as UniversalFileDownloadTask},
};
use tokio::runtime::Handle as TokioHandle;
use uuid::Uuid;

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_pause_with_lower_byte_count_flips_to_paused() {
    let task = UniversalFileDownloadTask::new(
        Uuid::new_v4(),
        "http://example.test/file".to_string(),
        PathBuf::from("/tmp/uzu-pause-regression-universal"),
        FileCheck::None,
        "test-manager".to_string(),
        Some(10_000_000),
        InternalDownloadState::NotDownloaded,
        FileDownloadState::downloading(1_000_000, 10_000_000),
        AsyncFetcherConfig::default(),
        TokioHandle::current(),
    );

    task.update_state_and_broadcast(FileDownloadState::paused(0, 10_000_000)).await;

    let state = task.state().await;
    assert!(
        matches!(state.phase, FileDownloadPhase::Paused),
        "expected Paused, got {:?}",
        state.phase
    );
}

#[cfg(target_vendor = "apple")]
#[tokio::test(flavor = "multi_thread")]
async fn test_apple_pause_with_lower_byte_count_flips_to_paused() {
    use download_manager::managers::apple::FileDownloadTask as AppleFileDownloadTask;

    let task = AppleFileDownloadTask::new(
        Uuid::new_v4(),
        "http://example.test/file".to_string(),
        PathBuf::from("/tmp/uzu-pause-regression-apple"),
        FileCheck::None,
        "test-manager".to_string(),
        Some(10_000_000),
        InternalDownloadState::NotDownloaded,
        FileDownloadState::downloading(1_000_000, 10_000_000),
        None,
        TokioHandle::current(),
    );

    task.update_state_and_broadcast(FileDownloadState::paused(0, 10_000_000)).await;

    let state = task.state().await;
    assert!(
        matches!(state.phase, FileDownloadPhase::Paused),
        "expected Paused, got {:?}",
        state.phase
    );
}
