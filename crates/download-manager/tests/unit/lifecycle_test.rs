use download_manager::{FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase};
use rstest::rstest;
use tokio::runtime::Handle as TokioHandle;

use crate::common::{MockRegistry, init_test_tracing, wait_for_phase};

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_download_fresh_completes(
    #[case] download_manager_type: FileDownloadManagerType,
) -> Result<(), Box<dyn std::error::Error>> {
    init_test_tracing();
    tracing::info!(?download_manager_type, "starting fresh download lifecycle test");
    let registry = MockRegistry::start().await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempfile::tempdir().unwrap();
    let destination = temp_dir.path().join(&tokenizer.file.name);

    let manager = <dyn FileDownloadManager>::new(download_manager_type, TokioHandle::current()).await.unwrap();
    let task = manager
        .file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await
        .unwrap();
    let mut progress = task.progress().await.unwrap();

    tracing::info!(destination = %destination.display(), "starting file download");
    task.download().await.unwrap();
    let state = wait_for_phase(&task, &mut progress, |phase| matches!(phase, FileDownloadPhase::Downloaded)).await;
    tracing::info!(?state, "file download reached terminal state");

    assert_eq!(state.downloaded_bytes, tokenizer.file.size as u64);
    assert_eq!(state.total_bytes, tokenizer.file.size as u64);
    assert_eq!(tokio::fs::read(&destination).await.unwrap(), tokenizer.bytes.to_vec());
    assert!(std::path::PathBuf::from(format!("{}.crc", destination.display())).is_file());
    Ok(())
}
