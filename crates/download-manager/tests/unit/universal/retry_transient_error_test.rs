use download_manager::{FileDownloadManagerType, create_download_manager};
use tokio::runtime::Handle as TokioHandle;

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::{DownloadTestContext, PhaseKind, wait_for_phase_kind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_retries_transient_server_error() {
    let context = DownloadTestContext::new(
        "tokenizer.json",
        RouteBehavior::RetryThenOk {
            failures: 1,
            status: 500,
        },
    )
    .await;
    let manager = create_download_manager(
        FileDownloadManagerType::Universal,
        Some("universal-retry".to_string()),
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
    let state = wait_for_phase_kind(&task, &mut progress, PhaseKind::Downloaded).await;
    context.assert_downloaded(&state).await;

    let statuses = context.server.records_snapshot().await.into_iter().map(|record| record.status).collect::<Vec<_>>();
    assert!(statuses.contains(&500), "mock server should have produced a retryable failure");
    assert!(statuses.contains(&200) || statuses.contains(&206), "download should retry and complete");
}
