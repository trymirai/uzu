use std::{path::PathBuf, sync::Arc};

use download_manager::{
    FileCheck, FileDownloadPhase, FileDownloadTask,
    backends::universal::{UniversalBackend, UniversalBackendContext},
    file_download_task_actor::{GenericFileDownloadTask, ProgressCounters, PublicProjection},
    reducer::InitialLifecycleState,
    traits::DownloadConfig,
};
use tokio::runtime::Handle as TokioHandle;
use uuid::Uuid;

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_pause_with_lower_byte_count_flips_to_paused() {
    let task = GenericFileDownloadTask::<UniversalBackend>::spawn(
        Arc::new(DownloadConfig {
            download_id: Uuid::new_v4(),
            source_url: "http://example.test/file".to_string(),
            destination: PathBuf::from("/tmp/uzu-pause-regression-universal-fsm"),
            file_check: FileCheck::None,
            expected_bytes: Some(10_000_000),
            manager_id: "test-manager".to_string(),
            manager_instance_id: Uuid::nil(),
        }),
        Arc::new(UniversalBackendContext::new(TokioHandle::current())),
        InitialLifecycleState::Paused {
            part_path: PathBuf::from("/tmp/uzu-pause-regression-universal-fsm.part"),
        },
        PublicProjection::None,
        ProgressCounters {
            downloaded_bytes: 0,
            total_bytes: 10_000_000,
        },
    )
    .unwrap();

    let state = task.state().await;
    assert!(matches!(state.phase, FileDownloadPhase::Paused), "expected Paused, got {:?}", state.phase);
}

#[cfg(target_vendor = "apple")]
#[tokio::test(flavor = "multi_thread")]
async fn test_apple_pause_with_lower_byte_count_flips_to_paused() {
    use download_manager::backends::apple::{AppleBackend, AppleBackendContext};

    let task = GenericFileDownloadTask::<AppleBackend>::spawn(
        Arc::new(DownloadConfig {
            download_id: Uuid::new_v4(),
            source_url: "http://example.test/file".to_string(),
            destination: PathBuf::from("/tmp/uzu-pause-regression-apple-fsm"),
            file_check: FileCheck::None,
            expected_bytes: Some(10_000_000),
            manager_id: "test-manager".to_string(),
            manager_instance_id: Uuid::nil(),
        }),
        Arc::new(AppleBackendContext::new(TokioHandle::current())),
        InitialLifecycleState::Paused {
            part_path: PathBuf::from("/tmp/uzu-pause-regression-apple-fsm.part"),
        },
        PublicProjection::None,
        ProgressCounters {
            downloaded_bytes: 0,
            total_bytes: 10_000_000,
        },
    )
    .unwrap();

    let state = task.state().await;
    assert!(matches!(state.phase, FileDownloadPhase::Paused), "expected Paused, got {:?}", state.phase);
}
