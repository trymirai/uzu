use download_manager::{FileDownloadManagerType, create_download_manager};
use tokio::{
    runtime::Handle as TokioHandle,
    time::{sleep as tokio_sleep, timeout as tokio_timeout},
};

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::{DownloadTestContext, PhaseKind, wait_for_phase_kind, wait_for_progress_bytes},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_pause_resume_completes_from_partial_state() {
    let context = DownloadTestContext::new(
        "model.safetensors",
        RouteBehavior::SlowChunks {
            chunk_size: 16 * 1024,
            delay_ms: 1,
        },
    )
    .await;
    let manager = create_download_manager(
        FileDownloadManagerType::Universal,
        Some("universal-range-resume".to_string()),
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
    let progress_state = wait_for_progress_bytes(&task, &mut progress, 64 * 1024).await;
    assert!(progress_state.downloaded_bytes > 0);
    task.pause().await.expect("failed to pause download");
    let paused_state = wait_for_phase_kind(&task, &mut progress, PhaseKind::Paused).await;
    assert!(paused_state.downloaded_bytes > 0);
    assert!(paused_state.downloaded_bytes <= paused_state.total_bytes);

    tokio_timeout(std::time::Duration::from_secs(2), async {
        while context.lock_path().exists() {
            tokio_sleep(std::time::Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("lock should be released after pause");
    tokio_sleep(std::time::Duration::from_millis(200)).await;

    task.download().await.expect("failed to resume download");
    let state = wait_for_phase_kind(&task, &mut progress, PhaseKind::Downloaded).await;
    context.assert_downloaded(&state).await;
}
