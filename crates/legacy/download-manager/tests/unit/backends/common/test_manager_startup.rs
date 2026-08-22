use chrono::Utc;
use download_manager::{
    FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase, compute_download_id,
    traits::DownloadConfig,
};
use kiban::rt::RuntimeHandle;
use tempfile::tempdir;

use crate::common::{MockRegistry, write_file_with_parents, write_recoverable_resume_artifact};

#[tokio::test(flavor = "multi_thread")]
async fn startup_keeps_same_stem_resume_artifacts_separate() -> Result<(), Box<dyn std::error::Error>> {
    let temporary_directory = tempdir()?;
    let binary_destination = temporary_directory.path().join("weights.bin");
    let safetensors_destination = temporary_directory.path().join("weights.safetensors");
    let binary_resume_artifact =
        DownloadConfig::resume_artifact_path_for(&binary_destination, compute_download_id(&binary_destination), "part");
    write_recoverable_resume_artifact(
        &binary_resume_artifact,
        &binary_destination,
        "http://example.invalid/weights.bin",
        FileCheck::None,
        Some(100),
        b"partial",
    )
    .await?;

    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    let binary_task = manager
        .file_download_task("http://example.invalid/weights.bin", &binary_destination, FileCheck::None, Some(100))
        .await?;
    let safetensors_task = manager
        .file_download_task(
            "http://example.invalid/weights.safetensors",
            &safetensors_destination,
            FileCheck::None,
            Some(100),
        )
        .await?;

    assert_eq!(binary_task.state().await.phase, FileDownloadPhase::Paused);
    assert_eq!(safetensors_task.state().await.phase, FileDownloadPhase::NotDownloaded);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_manager_startup_valid_existing_file_is_downloaded() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let expected_crc = served_file.crc32c()?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let receipt_path = DownloadConfig::integrity_receipt_path_for(&destination, compute_download_id(&destination));
    let resume_artifact =
        DownloadConfig::resume_artifact_path_for(&destination, compute_download_id(&destination), "part");
    tokio::fs::write(&destination, served_file.bytes.as_ref()).await?;
    write_file_with_parents(&resume_artifact, b"partial").await?;

    let task = manager_task(
        &served_file.file.url,
        &destination,
        FileCheck::CRC(expected_crc.clone()),
        Some(served_file.file.size as u64),
    )
    .await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    assert!(!resume_artifact.exists());
    let crc_receipt: serde_json::Value = serde_json::from_str(&tokio::fs::read_to_string(receipt_path).await?)?;
    assert_eq!(crc_receipt["version"].as_u64(), Some(1));
    assert_eq!(crc_receipt["crc"].as_str(), Some(expected_crc.as_str()));
    assert_eq!(crc_receipt["file_size"].as_u64(), Some(served_file.file.size as u64));
    assert!(crc_receipt["modified_unix_seconds"].as_u64().is_some());
    assert!(crc_receipt["modified_nanos"].as_u64().is_some());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_manager_startup_rejects_legacy_crc_cache_after_file_changes() -> Result<(), Box<dyn std::error::Error>> {
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

    assert_eq!(task.state().await.phase, FileDownloadPhase::NotDownloaded);
    assert!(!destination.exists());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_manager_startup_revalidates_stale_metadata_crc_cache() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let expected_crc = served_file.crc32c()?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let crc_path = std::path::PathBuf::from(format!("{}.crc", destination.display()));
    let mut stale_destination_bytes = served_file.bytes.to_vec();
    stale_destination_bytes[0] = stale_destination_bytes[0].wrapping_add(1);
    tokio::fs::write(&destination, stale_destination_bytes).await?;
    tokio::fs::write(
        &crc_path,
        serde_json::to_vec(&serde_json::json!({
            "version": 1,
            "crc": expected_crc.clone(),
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
    let resume_artifact =
        DownloadConfig::resume_artifact_path_for(&destination, compute_download_id(&destination), "part");
    let lock_path = std::env::temp_dir()
        .join("uzu-download-manager")
        .join("locks")
        .join(format!("{}.lock", compute_download_id(&destination)));
    tokio::fs::write(&destination, b"corrupt").await?;
    tokio::fs::write(&crc_path, served_file.crc32c()?).await?;
    write_file_with_parents(&resume_artifact, b"partial").await?;
    tokio::fs::create_dir_all(lock_path.parent().expect("lock path has a parent")).await?;
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
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    manager.file_download_task(source_url, destination, file_check, expected_bytes).await
}
