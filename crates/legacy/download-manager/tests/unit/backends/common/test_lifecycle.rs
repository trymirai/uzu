use download_manager::{FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase};
use hex::encode as hex_encode;
use kiban::rt::RuntimeHandle;
use rstest::rstest;
use sha2::{Digest, Sha256};
use tempfile::tempdir;
use tokio::fs::read as tokio_read;

use crate::common::{MockRegistry, wait_for_phase};

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_download_fresh_completes(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempdir().unwrap();
    let destination = temp_dir.path().join(&tokenizer.file.name);

    let manager = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await.unwrap();
    let task = manager
        .file_download_task(
            (&tokenizer.file.url).into(),
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await
        .unwrap();
    let mut progress = task.progress().await.unwrap();

    task.download().await.unwrap();
    let state = wait_for_phase(&task, &mut progress, |phase| matches!(phase, FileDownloadPhase::Downloaded)).await;

    assert_eq!(state.downloaded_bytes, tokenizer.file.size as u64);
    assert_eq!(state.total_bytes, tokenizer.file.size as u64);
    assert_eq!(tokio_read(&destination).await.unwrap(), tokenizer.bytes.to_vec());
    assert!(destination.with_added_extension("integrity").is_file());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_sha256_download_completes() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join("tokenizer.json");
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    let check = FileCheck::Sha256(hex_encode(Sha256::digest(&tokenizer.bytes)));
    let task = manager
        .file_download_task(
            (&tokenizer.file.url).into(),
            &destination,
            check,
            Some(tokenizer.file.size as u64),
        )
        .await?;
    let mut progress = task.progress().await?;

    task.download().await?;
    wait_for_phase(&task, &mut progress, |phase| matches!(phase, FileDownloadPhase::Downloaded)).await;

    assert_eq!(tokio_read(destination).await?, tokenizer.bytes.as_ref());
    Ok(())
}
