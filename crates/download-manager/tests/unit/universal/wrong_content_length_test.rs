use download_manager::{FileDownloadManagerType, create_download_manager};
use tokio::runtime::Handle as TokioHandle;

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::{DownloadTestContext, PhaseKind, error_message, wait_for_phase_kind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_wrong_content_length_fails() {
    let context = DownloadTestContext::new(
        "tokenizer.json",
        RouteBehavior::WrongContentLength,
    )
    .await;
    let manager = create_download_manager(
        FileDownloadManagerType::Universal,
        Some("universal-wrong-length".to_string()),
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
    let state = wait_for_phase_kind(&task, &mut progress, PhaseKind::Error).await;
    let message = error_message(state);
    assert!(!message.is_empty(), "wrong content length should surface an error");
}
