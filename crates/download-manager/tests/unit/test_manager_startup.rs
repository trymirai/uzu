use std::sync::Arc;

use chrono::Utc;
use download_manager::{FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase};
use rstest::rstest;
use tempfile::tempdir;
use tokio::runtime::Handle as TokioHandle;

use crate::common::MockRegistry;

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_manager_constructs_actor_backed_task_from_reducer(
    #[case] download_manager_type: FileDownloadManagerType,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join(&served_file.file.name);
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

    let state = task.state().await;

    assert_eq!(state.phase, FileDownloadPhase::NotDownloaded);
    assert_eq!(state.total_bytes, served_file.file.size as u64);
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_manager_returns_cached_task_for_same_download(
    #[case] download_manager_type: FileDownloadManagerType,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join(&served_file.file.name);
    let manager = <dyn FileDownloadManager>::new(download_manager_type, TokioHandle::current()).await.unwrap();
    let first_task = manager
        .file_download_task(
            &served_file.file.url,
            &destination,
            FileCheck::None,
            Some(served_file.file.size as u64),
        )
        .await
        .unwrap();
    let second_task = manager
        .file_download_task(
            &served_file.file.url,
            &destination,
            FileCheck::None,
            Some(served_file.file.size as u64),
        )
        .await
        .unwrap();

    assert!(Arc::ptr_eq(&first_task, &second_task));
    assert_eq!(manager.get_all_file_tasks().await.unwrap().len(), 1);
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_manager_applies_reducer_actions_for_valid_existing_file(
    #[case] download_manager_type: FileDownloadManagerType,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join(&served_file.file.name);
    let resume_artifact = destination.with_extension(resume_artifact_extension(download_manager_type));
    tokio::fs::write(&destination, served_file.bytes.as_ref()).await.unwrap();
    tokio::fs::write(&resume_artifact, b"partial").await.unwrap();

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

    let state = task.state().await;
    assert_eq!(state.phase, FileDownloadPhase::Downloaded);
    assert!(!resume_artifact.exists());
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_manager_uses_crc_observation_and_deletes_invalid_file(
    #[case] download_manager_type: FileDownloadManagerType,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join(&served_file.file.name);
    tokio::fs::write(&destination, b"corrupt").await.unwrap();

    let manager = <dyn FileDownloadManager>::new(download_manager_type, TokioHandle::current()).await.unwrap();
    let task = manager
        .file_download_task(
            &served_file.file.url,
            &destination,
            FileCheck::CRC(served_file.crc32c()?),
            Some("corrupt".len() as u64),
        )
        .await
        .unwrap();

    let state = task.state().await;
    assert_eq!(state.phase, FileDownloadPhase::NotDownloaded);
    assert!(!destination.exists());
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_manager_writes_crc_cache_from_action_plan(
    #[case] download_manager_type: FileDownloadManagerType,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let expected_crc = served_file.crc32c()?;
    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join(&served_file.file.name);
    let crc_path = std::path::PathBuf::from(format!("{}.crc", destination.display()));
    tokio::fs::write(&destination, served_file.bytes.as_ref()).await.unwrap();

    let manager = <dyn FileDownloadManager>::new(download_manager_type, TokioHandle::current()).await.unwrap();
    let task = manager
        .file_download_task(
            &served_file.file.url,
            &destination,
            FileCheck::CRC(expected_crc.clone()),
            Some(served_file.file.size as u64),
        )
        .await
        .unwrap();

    let state = task.state().await;
    assert_eq!(state.phase, FileDownloadPhase::Downloaded);
    assert_eq!(tokio::fs::read_to_string(crc_path).await.unwrap(), expected_crc);
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_manager_foreign_lock_suppresses_startup_cleanup_actions(
    #[case] download_manager_type: FileDownloadManagerType,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join(&served_file.file.name);
    let crc_path = std::path::PathBuf::from(format!("{}.crc", destination.display()));
    let resume_artifact = destination.with_extension(resume_artifact_extension(download_manager_type));
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

    let manager = <dyn FileDownloadManager>::new(download_manager_type, TokioHandle::current()).await.unwrap();
    let task = manager
        .file_download_task(
            &served_file.file.url,
            &destination,
            FileCheck::CRC(served_file.crc32c()?),
            Some(served_file.file.size as u64),
        )
        .await
        .unwrap();

    let state = task.state().await;
    assert!(matches!(state.phase, FileDownloadPhase::LockedByOther(_)));
    assert!(destination.exists());
    assert!(crc_path.exists());
    assert!(resume_artifact.exists());
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_manager_revalidates_matching_crc_cache(#[case] download_manager_type: FileDownloadManagerType) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let expected_crc = served_file.crc32c()?;
    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join(&served_file.file.name);
    let crc_path = std::path::PathBuf::from(format!("{}.crc", destination.display()));
    tokio::fs::write(&destination, b"corrupt").await?;
    tokio::fs::write(&crc_path, expected_crc).await?;

    let manager = <dyn FileDownloadManager>::new(download_manager_type, TokioHandle::current()).await.unwrap();
    let task = manager
        .file_download_task(
            &served_file.file.url,
            &destination,
            FileCheck::CRC(served_file.crc32c()?),
            Some(served_file.file.size as u64),
        )
        .await
        .unwrap();

    let state = task.state().await;
    assert_eq!(state.phase, FileDownloadPhase::NotDownloaded);
    assert!(!destination.exists());
    Ok(())
}

fn resume_artifact_extension(download_manager_type: FileDownloadManagerType) -> &'static str {
    match download_manager_type {
        FileDownloadManagerType::Universal => "part",
        FileDownloadManagerType::Apple => "resume_data",
    }
}
