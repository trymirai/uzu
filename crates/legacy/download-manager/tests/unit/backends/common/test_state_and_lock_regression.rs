use std::{error::Error, sync::Arc};

use download_manager::{DownloadError, FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase};
use kiban::rt::RuntimeHandle;
use rstest::rstest;
use tempfile::tempdir;

use crate::common::{Behavior, MockRegistry, wait_for_phase};

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_task_creation_returns_same_task(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn Error>> {
    let registry = MockRegistry::start().await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempdir().unwrap();
    let destination = temp_dir.path().join(&tokenizer.file.name);
    let manager = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await.unwrap();

    let (first, second) = tokio::join!(
        manager.file_download_task(
            (&tokenizer.file.url).into(),
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        ),
        manager.file_download_task(
            (&tokenizer.file.url).into(),
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        ),
    );
    let first = first.unwrap();
    let second = second.unwrap();

    assert!(Arc::ptr_eq(&first, &second));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn conflicting_task_configuration_is_rejected() -> Result<(), Box<dyn Error>> {
    let registry = MockRegistry::start().await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempdir().unwrap();
    let destination = temp_dir.path().join(&tokenizer.file.name);
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;

    manager
        .file_download_task(
            (&tokenizer.file.url).into(),
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await?;
    let different_url = manager
        .file_download_task(
            "http://example.invalid/different-url".into(),
            &destination,
            FileCheck::None,
            Some(tokenizer.file.size as u64),
        )
        .await;
    let different_size = manager
        .file_download_task(
            (&tokenizer.file.url).into(),
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64 + 1),
        )
        .await;

    assert!(matches!(different_url, Err(DownloadError::ConflictingConfig(_))));
    assert!(matches!(different_size, Err(DownloadError::ConflictingConfig(_))));
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn separate_managers_in_same_process_cannot_share_destination_lock(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempdir().unwrap();
    let destination = temp_dir.path().join(&tokenizer.file.name);

    let manager_a = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await.unwrap();
    let task_a = manager_a
        .file_download_task(
            (&tokenizer.file.url).into(),
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await
        .unwrap();
    let mut progress_a = task_a.progress().await.unwrap();
    task_a.download().await.unwrap();
    wait_for_phase(&task_a, &mut progress_a, |phase| matches!(phase, FileDownloadPhase::Downloading)).await;

    let manager_b = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await.unwrap();
    let task_b = manager_b
        .file_download_task(
            (&tokenizer.file.url).into(),
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await
        .unwrap();

    assert!(matches!(task_b.state().await.phase, FileDownloadPhase::LockedByOther(_)));

    drop(task_b);
    drop(manager_b);

    let progress_a_after_b_dropped = task_a.state().await.phase;
    assert!(!matches!(progress_a_after_b_dropped, FileDownloadPhase::Error(_)));
    Ok(())
}
