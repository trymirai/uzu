use download_manager::{FileCheck, FileDownloadManagerType, FileDownloadPhase, create_download_manager};
use tokio::runtime::Handle as TokioHandle;

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::{DownloadTestContext, PhaseKind, wait_for_phase_kind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_lock_file_prevents_concurrent_destination() {
    let context = DownloadTestContext::new(
        "model.safetensors",
        RouteBehavior::StallAt {
            byte_offset: 64 * 1024,
        },
    )
    .await;
    let manager = create_download_manager(
        FileDownloadManagerType::Apple,
        Some("apple-lock-owner".to_string()),
        TokioHandle::current(),
    )
    .await
    .expect("failed to create download manager");
    let task = manager
        .file_download_task(&context.payload.file.url, &context.destination, context.file_check(), context.file_size())
        .await
        .expect("failed to create file download task");
    let mut progress = task.progress().await.expect("progress stream should open");

    task.download().await.expect("failed to start download");
    context.wait_for_bytes(64 * 1024).await;
    assert!(context.lock_path().exists(), "download should acquire a destination lock");

    let other_manager = create_download_manager(
        FileDownloadManagerType::Apple,
        Some("apple-lock-other-manager".to_string()),
        TokioHandle::current(),
    )
    .await
    .expect("failed to create second manager");
    let other_task = other_manager
        .file_download_task(
            &context.payload.file.url,
            &context.destination,
            FileCheck::CRC(context.payload.crc32c()),
            Some(context.payload.file.size as u64),
        )
        .await
        .expect("failed to create second task");

    let other_state = other_task.state().await;
    assert!(matches!(other_state.phase, FileDownloadPhase::LockedByOther(_)));
    task.cancel().await.expect("failed to cancel download");
    wait_for_phase_kind(&task, &mut progress, PhaseKind::NotDownloaded).await;
}
