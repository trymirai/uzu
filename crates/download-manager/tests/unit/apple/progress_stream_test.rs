use std::time::Duration;

use download_manager::{FileDownloadManagerType, FileDownloadPhase, create_download_manager};
use tokio::{runtime::Handle as TokioHandle, time::timeout as tokio_timeout};
use tokio_stream::StreamExt;

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::DownloadTestContext,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_progress_stream_reports_downloaded() {
    let context = DownloadTestContext::new(
        "tokenizer.json",
        RouteBehavior::SlowChunks {
            chunk_size: 16 * 1024,
            delay_ms: 1,
        },
    )
    .await;
    let manager = create_download_manager(
        FileDownloadManagerType::Apple,
        Some("apple-broadcast".to_string()),
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
    let observed_downloaded = tokio_timeout(Duration::from_secs(20), async {
        while let Some(update) = progress.next().await {
            if let Ok(state) = update {
                if matches!(state.phase, FileDownloadPhase::Downloaded) {
                    return true;
                }
            }
        }
        false
    })
    .await
    .unwrap_or(false);
    assert!(observed_downloaded, "progress stream should publish Downloaded");
    let state = task.state().await;
    context.assert_downloaded(&state).await;
}
