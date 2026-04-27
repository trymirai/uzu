use download_manager::{FileDownloadManagerType, FileDownloadPhase, create_download_manager};
use tokio::runtime::Handle as TokioHandle;

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::{DownloadTestContext, PhaseKind, wait_for_phase_kind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_forced_disconnect_fails_or_retries_cleanly() {
    let context = DownloadTestContext::new(
        "config.json",
        RouteBehavior::DisconnectAt {
            byte_offset: 1024,
        },
    )
    .await;
    let manager = create_download_manager(
        FileDownloadManagerType::Universal,
        Some("universal-disconnect".to_string()),
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
    context.wait_for_bytes(1024).await;
    task.cancel().await.expect("failed to cancel download");
    let state = wait_for_phase_kind(&task, &mut progress, PhaseKind::NotDownloaded).await;
    assert!(matches!(state.phase, FileDownloadPhase::NotDownloaded));
}
