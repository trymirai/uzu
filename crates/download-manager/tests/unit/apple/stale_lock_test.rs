use chrono::{Duration as ChronoDuration, Utc};
use download_manager::{FileDownloadManagerType, create_download_manager};
use serde_json::to_vec;
use tokio::{fs::write as tokio_write, runtime::Handle as TokioHandle};

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::{DownloadTestContext, PhaseKind, wait_for_phase_kind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_stale_lock_can_be_reacquired() {
    let context = DownloadTestContext::new(
        "tokenizer.json",
        RouteBehavior::Normal,
    )
    .await;
    let manager = create_download_manager(
        FileDownloadManagerType::Apple,
        Some("apple-stale-lock".to_string()),
        TokioHandle::current(),
    )
    .await
    .expect("failed to create download manager");
    let lock_path = context.lock_path();
    let stale_lock = serde_json::json!({
        "manager_id": "other-manager",
        "acquired_at": Utc::now() - ChronoDuration::hours(2),
        "process_id": 999_999_u32,
    });
    tokio_write(&lock_path, to_vec(&stale_lock).expect("failed to serialize lock"))
        .await
        .expect("failed to seed stale lock");

    let task = manager
        .file_download_task(&context.payload.file.url, &context.destination, context.file_check(), context.file_size())
        .await
        .expect("failed to create file download task");
    let mut progress = task.progress().await.expect("progress stream should open");

    task.download().await.expect("stale lock should be taken over");
    let state = wait_for_phase_kind(&task, &mut progress, PhaseKind::Downloaded).await;
    context.assert_downloaded(&state).await;
    context.expect_no_temp_artifacts().await;
}
