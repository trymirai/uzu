use chrono::Utc;
use download_manager::{FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase};
use tempfile::tempdir;
use tokio::runtime::Handle as TokioHandle;

use crate::common::MockRegistry;

#[tokio::test(flavor = "multi_thread")]
async fn test_manager_startup_valid_existing_file_is_downloaded() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let expected_crc = served_file.crc32c()?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let crc_path = std::path::PathBuf::from(format!("{}.crc", destination.display()));
    let resume_artifact = destination.with_extension("part");
    tokio::fs::write(&destination, served_file.bytes.as_ref()).await?;
    tokio::fs::write(&resume_artifact, b"partial").await?;

    let task = manager_task(&served_file.file.url, &destination, FileCheck::CRC(expected_crc.clone()), Some(served_file.file.size as u64)).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    assert!(!resume_artifact.exists());
    assert_eq!(tokio::fs::read_to_string(crc_path).await?, expected_crc);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_manager_startup_uses_matching_crc_cache_fast_path() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let expected_crc = served_file.crc32c()?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let crc_path = std::path::PathBuf::from(format!("{}.crc", destination.display()));
    let mut stale_destination_bytes = served_file.bytes.to_vec();
    stale_destination_bytes[0] = stale_destination_bytes[0].wrapping_add(1);
    tokio::fs::write(&destination, stale_destination_bytes).await?;
    tokio::fs::write(&crc_path, &expected_crc).await?;

    let task = manager_task(
        &served_file.file.url,
        &destination,
        FileCheck::CRC(expected_crc),
        Some(served_file.file.size as u64),
    )
    .await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_manager_startup_revalidates_crc_cache() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let expected_crc = served_file.crc32c()?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let crc_path = std::path::PathBuf::from(format!("{}.crc", destination.display()));
    tokio::fs::write(&destination, b"corrupt").await?;
    tokio::fs::write(&crc_path, &expected_crc).await?;

    let task = manager_task(&served_file.file.url, &destination, FileCheck::CRC(expected_crc), Some(served_file.file.size as u64)).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::NotDownloaded);
    assert!(!destination.exists());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_manager_startup_directory_destination_is_not_downloaded() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    tokio::fs::create_dir(&destination).await?;

    let task = manager_task(&served_file.file.url, &destination, FileCheck::None, None).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::NotDownloaded);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_manager_startup_foreign_lock_preserves_files() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let crc_path = std::path::PathBuf::from(format!("{}.crc", destination.display()));
    let resume_artifact = destination.with_extension("part");
    let lock_path = std::path::PathBuf::from(format!("{}.lock", destination.display()));
    tokio::fs::write(&destination, b"corrupt").await?;
    tokio::fs::write(&crc_path, served_file.crc32c()?).await?;
    tokio::fs::write(&resume_artifact, b"partial").await?;
    tokio::fs::write(
        &lock_path,
        serde_json::to_vec(&serde_json::json!({
            "manager_id": "foreign-manager",
            "acquired_at": Utc::now(),
            "process_id": std::process::id(),
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
    assert!(crc_path.exists());
    assert!(resume_artifact.exists());
    Ok(())
}

async fn manager_task(
    source_url: &str,
    destination: &std::path::Path,
    file_check: FileCheck,
    expected_bytes: Option<u64>,
) -> Result<std::sync::Arc<dyn download_manager::FileDownloadTask>, download_manager::DownloadError> {
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, TokioHandle::current()).await?;
    manager.file_download_task(source_url, destination, file_check, expected_bytes).await
}
