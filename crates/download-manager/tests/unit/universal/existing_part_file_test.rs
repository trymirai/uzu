use download_manager::{FileDownloadManagerType, FileDownloadPhase, create_download_manager};
use tokio::runtime::Handle as TokioHandle;

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::DownloadTestContext,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_existing_part_file_resumes_from_partial_size() {
    let context =
        DownloadTestContext::new_with_existing_part_size("model.safetensors", RouteBehavior::Normal, Some(64 * 1024))
            .await;
    let manager = create_download_manager(
        FileDownloadManagerType::Universal,
        Some("universal-existing-part".to_string()),
        TokioHandle::current(),
    )
    .await
    .expect("failed to create download manager");
    let task = manager
        .file_download_task(&context.payload.file.url, &context.destination, context.file_check(), context.file_size())
        .await
        .expect("failed to create file download task");

    let state = task.state().await;
    assert!(matches!(state.phase, FileDownloadPhase::Paused));
    assert_eq!(state.downloaded_bytes, 64 * 1024);
}
