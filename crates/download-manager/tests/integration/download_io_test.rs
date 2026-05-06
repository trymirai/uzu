use std::time::Duration;

use download_manager::{FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase};
use mock_registry::MockRegistry;
use rstest::rstest;
use tempfile::tempdir;
use tokio::{runtime::Handle as TokioHandle, time::timeout};

use crate::common::init_test_tracing;

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_download_manager_downloads_mock_registry_file_to_destination(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    init_test_tracing();
    tracing::info!(?download_manager_type, "starting integration file download test");
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join(&served_file.file.name);
    let manager = <dyn FileDownloadManager>::new(download_manager_type, TokioHandle::current()).await?;
    let task = manager
        .file_download_task(
            &served_file.file.url,
            &destination,
            FileCheck::CRC(served_file.crc32c()?),
            Some(served_file.file.size as u64),
        )
        .await?;

    task.download().await?;
    timeout(Duration::from_secs(10), task.wait()).await?;
    tracing::info!(destination = %destination.display(), "download wait completed");

    let state = task.state().await;
    assert_eq!(state.phase, FileDownloadPhase::Downloaded);
    assert_eq!(state.downloaded_bytes, served_file.file.size as u64);
    assert_eq!(state.total_bytes, served_file.file.size as u64);
    assert_eq!(tokio::fs::read(&destination).await?, served_file.bytes.to_vec());
    assert!(std::path::PathBuf::from(format!("{}.crc", destination.display())).is_file());

    Ok(())
}
