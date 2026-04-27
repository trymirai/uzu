use download_manager::{FileDownloadManagerType, create_download_manager};
use tokio::{
    runtime::Handle as TokioHandle,
    time::{Duration, sleep as tokio_sleep, timeout as tokio_timeout},
};
use uuid::Uuid;

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::{
        DownloadTestContext, PhaseKind, download_manager_test_name, wait_for_phase_kind, wait_for_progress_bytes,
    },
};

pub async fn run_fresh_download_scenario(download_manager_type: FileDownloadManagerType) {
    let context = DownloadTestContext::new("tokenizer.json", RouteBehavior::Normal).await;
    let manager_name = format!("{}_fresh_{}", download_manager_test_name(download_manager_type), Uuid::new_v4());
    let manager = create_download_manager(download_manager_type, Some(manager_name), TokioHandle::current())
        .await
        .expect("failed to create download manager");
    let task = manager
        .file_download_task(&context.payload.file.url, &context.destination, context.file_check(), context.file_size())
        .await
        .expect("failed to create file download task");
    let mut progress = task.progress().await.expect("progress stream should open");

    task.download().await.expect("failed to start download");
    let state = wait_for_phase_kind(&task, &mut progress, PhaseKind::Downloaded).await;
    context.assert_downloaded(&state).await;
    context.expect_no_temp_artifacts().await;
}

pub async fn run_cancel_redownload_scenario(download_manager_type: FileDownloadManagerType) {
    let context = DownloadTestContext::new(
        "model.safetensors",
        RouteBehavior::SlowChunks {
            chunk_size: 16 * 1024,
            delay_ms: 1,
        },
    )
    .await;
    let manager_name = format!("{}_cancel_{}", download_manager_test_name(download_manager_type), Uuid::new_v4());
    let manager = create_download_manager(download_manager_type, Some(manager_name), TokioHandle::current())
        .await
        .expect("failed to create download manager");
    let task = manager
        .file_download_task(&context.payload.file.url, &context.destination, context.file_check(), context.file_size())
        .await
        .expect("failed to create file download task");
    let mut progress = task.progress().await.expect("progress stream should open");

    task.download().await.expect("failed to start download");
    let progress_state = wait_for_progress_bytes(&task, &mut progress, 64 * 1024).await;
    assert!(progress_state.downloaded_bytes > 0, "download should report positive progress before cancel");
    task.cancel().await.expect("failed to cancel download");
    wait_for_phase_kind(&task, &mut progress, PhaseKind::NotDownloaded).await;

    task.download().await.expect("failed to resume download");
    let state = wait_for_phase_kind(&task, &mut progress, PhaseKind::Downloaded).await;
    context.assert_downloaded(&state).await;
}

pub async fn run_pause_resume_scenario(download_manager_type: FileDownloadManagerType) {
    let context = DownloadTestContext::new(
        "model.safetensors",
        RouteBehavior::SlowChunks {
            chunk_size: 16 * 1024,
            delay_ms: 1,
        },
    )
    .await;
    let manager_name = format!("{}_pause_{}", download_manager_test_name(download_manager_type), Uuid::new_v4());
    let manager = create_download_manager(download_manager_type, Some(manager_name), TokioHandle::current())
        .await
        .expect("failed to create download manager");
    let task = manager
        .file_download_task(&context.payload.file.url, &context.destination, context.file_check(), context.file_size())
        .await
        .expect("failed to create file download task");
    let mut progress = task.progress().await.expect("progress stream should open");

    task.download().await.expect("failed to start download");
    let progress_state = wait_for_progress_bytes(&task, &mut progress, 64 * 1024).await;
    assert!(progress_state.downloaded_bytes > 0, "download should report positive progress before pause");
    task.pause().await.expect("failed to pause download");
    let paused_state = wait_for_phase_kind(&task, &mut progress, PhaseKind::Paused).await;
    if download_manager_type == FileDownloadManagerType::Apple {
        assert!(paused_state.downloaded_bytes > 0, "Apple pause should preserve positive progress");
    }

    tokio_timeout(Duration::from_secs(2), async {
        while context.lock_path().exists() {
            tokio_sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("lock should be released after pause");
    tokio_sleep(Duration::from_millis(200)).await;

    task.download().await.expect("failed to resume download");
    let state = wait_for_phase_kind(&task, &mut progress, PhaseKind::Downloaded).await;
    context.assert_downloaded(&state).await;
}
