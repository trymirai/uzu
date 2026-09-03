use std::{error::Error, path::Path, process::id, sync::Arc};

use chrono::Utc;
use download_manager::{
    DownloadError, FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase, FileDownloadTask,
};
use kiban::rt::RuntimeHandle;
use serde_json::{json, to_vec};
use tempfile::tempdir;
use tokio::fs::{create_dir as tokio_create_dir, write as tokio_write};

use crate::common::MockRegistry;

#[tokio::test(flavor = "multi_thread")]
async fn test_manager_startup_valid_existing_file_is_downloaded() -> Result<(), Box<dyn Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let expected_crc = served_file.crc32c()?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let integrity_path = destination.with_added_extension("integrity");
    let resume_artifact = destination.with_added_extension("part");
    tokio_write(&destination, served_file.bytes.as_ref()).await?;
    tokio_write(&resume_artifact, b"partial").await?;

    let task = manager_task(
        &served_file.file.url,
        &destination,
        FileCheck::CRC(expected_crc.clone()),
        Some(served_file.file.size as u64),
    )
    .await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    assert!(!resume_artifact.exists());
    assert!(integrity_path.is_file());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_manager_startup_revalidates_stale_integrity_receipt() -> Result<(), Box<dyn Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let expected_crc = served_file.crc32c()?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let integrity_path = destination.with_added_extension("integrity");
    let mut stale_destination_bytes = served_file.bytes.to_vec();
    stale_destination_bytes[0] = stale_destination_bytes[0].wrapping_add(1);
    tokio_write(&destination, stale_destination_bytes).await?;
    tokio_write(
        &integrity_path,
        to_vec(&json!({
            "version": 1,
            "file_check": { "CRC": expected_crc.clone() },
            "file_size": served_file.file.size,
            "modified_unix_seconds": 0,
            "modified_nanos": 0,
        }))?,
    )
    .await?;

    let task = manager_task(
        &served_file.file.url,
        &destination,
        FileCheck::CRC(expected_crc),
        Some(served_file.file.size as u64),
    )
    .await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::NotDownloaded);
    assert!(!destination.exists());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_manager_startup_directory_destination_is_not_downloaded() -> Result<(), Box<dyn Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    tokio_create_dir(&destination).await?;

    let task = manager_task(&served_file.file.url, &destination, FileCheck::None, None).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::NotDownloaded);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_manager_startup_foreign_lock_preserves_files() -> Result<(), Box<dyn Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let integrity_path = destination.with_added_extension("integrity");
    let resume_artifact = destination.with_added_extension("part");
    let lock_path = destination.with_added_extension("lock");
    tokio_write(&destination, b"corrupt").await?;
    tokio_write(&integrity_path, served_file.crc32c()?).await?;
    tokio_write(&resume_artifact, b"partial").await?;
    tokio_write(
        &lock_path,
        to_vec(&json!({
            "manager_id": "foreign-manager",
            "acquired_at": Utc::now(),
            "process_id": id(),
        }))?,
    )
    .await?;

    let task = manager_task(
        &served_file.file.url,
        &destination,
        FileCheck::CRC(served_file.crc32c()?),
        Some(served_file.file.size as u64),
    )
    .await?;

    assert!(matches!(task.state().await.phase, FileDownloadPhase::LockedByOther(_)));
    assert!(destination.exists());
    assert!(integrity_path.exists());
    assert!(resume_artifact.exists());
    Ok(())
}

async fn manager_task(
    source_url: &str,
    destination: &Path,
    file_check: FileCheck,
    expected_bytes: Option<u64>,
) -> Result<Arc<dyn FileDownloadTask>, DownloadError> {
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    manager.file_download_task(source_url.into(), destination, file_check, expected_bytes).await
}
