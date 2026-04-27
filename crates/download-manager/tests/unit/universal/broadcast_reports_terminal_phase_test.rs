use std::time::Duration;

use download_manager::{FileDownloadManagerType, FileDownloadPhase, create_download_manager};
use tokio::{
    fs::read as tokio_read,
    runtime::Handle as TokioHandle,
    time::timeout as tokio_timeout,
};
use tokio_stream::StreamExt;

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::DownloadTestContext,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_broadcast_reports_terminal_phase() {
    let context = DownloadTestContext::new(
        "tokenizer.json",
        RouteBehavior::Normal,
    )
    .await;
    let manager = create_download_manager(
        FileDownloadManagerType::Universal,
        Some("universal-broadcast".to_string()),
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
    let observed_downloaded = tokio_timeout(Duration::from_secs(2), async {
        while let Some(Ok(state)) = progress.next().await {
            if matches!(state.phase, FileDownloadPhase::Downloaded) {
                return true;
            }
        }
        false
    })
    .await
    .unwrap_or(false);
    assert!(observed_downloaded, "progress stream should publish Downloaded");
    assert_eq!(
        tokio_read(&context.destination).await.expect("downloaded file missing"),
        context.payload.bytes.to_vec()
    );
}
