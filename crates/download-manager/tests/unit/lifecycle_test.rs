use download_manager::{FileDownloadManagerType, create_download_manager};
use rstest::rstest;
use tokio::runtime::Handle as TokioHandle;

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::{DownloadTestContext, PhaseKind, wait_for_phase_kind},
};

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_download_fresh_completes(#[case] download_manager_type: FileDownloadManagerType) {
    let context = DownloadTestContext::new("tokenizer.json", RouteBehavior::Normal).await;
    let manager = create_download_manager(download_manager_type, None, TokioHandle::current()).await.unwrap();
    let task = manager
        .file_download_task(&context.payload.file.url, &context.destination, context.file_check(), context.file_size())
        .await
        .unwrap();
    let mut progress = task.progress().await.unwrap();

    task.download().await.unwrap();
    let state = wait_for_phase_kind(&task, &mut progress, PhaseKind::Downloaded).await;
    context.assert_downloaded(&state).await;
}
