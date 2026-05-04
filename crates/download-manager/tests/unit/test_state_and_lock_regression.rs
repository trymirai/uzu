use std::sync::Arc;

use download_manager::{FileCheck, FileDownloadManager, FileDownloadManagerType};
use rstest::rstest;
use tokio::runtime::Handle as TokioHandle;

use crate::common::MockRegistry;

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_task_creation_returns_same_task(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempfile::tempdir().unwrap();
    let destination = temp_dir.path().join(&tokenizer.file.name);
    let manager = <dyn FileDownloadManager>::new(download_manager_type, TokioHandle::current())
        .await
        .unwrap();

    let (first, second) = tokio::join!(
        manager.file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        ),
        manager.file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        ),
    );
    let first = first.unwrap();
    let second = second.unwrap();

    assert!(
        Arc::ptr_eq(&first, &second),
        "same URL and destination should return the same task object"
    );
    Ok(())
}
