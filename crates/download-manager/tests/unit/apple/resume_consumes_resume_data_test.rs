use download_manager::{FileDownloadManagerType, create_download_manager};
use tokio::runtime::Handle as TokioHandle;

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::{DownloadTestContext, PhaseKind, wait_for_phase_kind, wait_for_progress_bytes},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_resume_consumes_resume_data_file() {
    let context = DownloadTestContext::new(
        "model.safetensors",
        RouteBehavior::SlowChunks {
            chunk_size: 16 * 1024,
            delay_ms: 1,
        },
    )
    .await;
    let manager = create_download_manager(
        FileDownloadManagerType::Apple,
        Some("apple-resume-consume".to_string()),
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
    wait_for_progress_bytes(&task, &mut progress, 64 * 1024).await;
    task.pause().await.expect("failed to pause download");
    wait_for_phase_kind(&task, &mut progress, PhaseKind::Paused).await;
    assert!(context.resume_data_path().exists(), "pause should persist resume data");

    task.download().await.expect("failed to resume download");
    assert!(!context.resume_data_path().exists(), "resume should consume persisted resume data");
    let state = wait_for_phase_kind(&task, &mut progress, PhaseKind::Downloaded).await;
    context.assert_downloaded(&state).await;
}
