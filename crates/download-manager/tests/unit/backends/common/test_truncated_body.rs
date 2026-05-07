use download_manager::{FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase};
use rstest::rstest;
use tokio::runtime::Handle as TokioHandle;

use crate::common::{Behavior, MockRegistry, wait_for_phase};

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn truncated_body_without_crc_fails_length_check(
    #[case] download_manager_type: FileDownloadManagerType,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::TRUNCATE_BODY).await?;
    let served_file = registry.file("config.json")?;
    let temp_dir = tempfile::tempdir().unwrap();
    let destination = temp_dir.path().join(&served_file.file.name);

    let manager = <dyn FileDownloadManager>::new(download_manager_type, TokioHandle::current()).await.unwrap();
    let task = manager
        .file_download_task(
            &served_file.file.url,
            &destination,
            FileCheck::None,
            Some(served_file.file.size as u64),
        )
        .await
        .unwrap();
    let mut progress = task.progress().await.unwrap();

    task.download().await.unwrap();
    let state = wait_for_phase(&task, &mut progress, |phase| {
        matches!(phase, FileDownloadPhase::Downloaded | FileDownloadPhase::Error(_))
    })
    .await;

    assert!(
        matches!(state.phase, FileDownloadPhase::Error(_)),
        "truncated body must surface as Error, got {:?}",
        state.phase,
    );
    Ok(())
}
