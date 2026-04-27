use download_manager::{FileDownloadManagerType, create_download_manager};
use tokio::runtime::Handle as TokioHandle;

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::{DownloadTestContext, PhaseKind, wait_for_phase_kind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_multi_connection_requests_byte_ranges() {
    let context = DownloadTestContext::new(
        "model.safetensors",
        RouteBehavior::Normal,
    )
    .await;
    let manager = create_download_manager(
        FileDownloadManagerType::Universal,
        Some("universal-range-split".to_string()),
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

    let range_requests = context
        .server
        .records_snapshot()
        .await
        .into_iter()
        .filter(|record| record.range.is_some())
        .collect::<Vec<_>>();
    assert!(!range_requests.is_empty(), "universal downloader should request byte ranges");
}
