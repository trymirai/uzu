use std::{path::Path, sync::Arc};

use base64::Engine;
use download_manager::{
    FileCheck, FileDownloadManager, FileDownloadPhase, backends::universal::UniversalDownloadManager,
};
use tempfile::tempdir;

#[tokio::test]
async fn test_universal_manager_constructs_actor_backed_task_from_reducer() {
    let manager = UniversalDownloadManager::new("test-manager".to_string());
    let task = manager
        .file_download_task(
            &"https://example.com/model.bin".to_string(),
            Path::new("missing-model.bin"),
            FileCheck::None,
            Some(120),
        )
        .await
        .unwrap();

    let state = task.state().await;

    assert_eq!(state.phase, FileDownloadPhase::NotDownloaded);
    assert_eq!(state.total_bytes, 120);
}

#[tokio::test]
async fn test_universal_manager_returns_cached_task_for_same_download() {
    let manager = UniversalDownloadManager::new("test-manager".to_string());
    let first_task = manager
        .file_download_task(
            &"https://example.com/model.bin".to_string(),
            Path::new("cached-model.bin"),
            FileCheck::None,
            Some(120),
        )
        .await
        .unwrap();
    let second_task = manager
        .file_download_task(
            &"https://example.com/model.bin".to_string(),
            Path::new("cached-model.bin"),
            FileCheck::None,
            Some(120),
        )
        .await
        .unwrap();

    assert!(Arc::ptr_eq(&first_task, &second_task));
    assert_eq!(manager.get_all_file_tasks().await.unwrap().len(), 1);
}

#[tokio::test]
async fn test_universal_manager_applies_reducer_actions_for_valid_existing_file() {
    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join("model.bin");
    let resume_artifact = destination.with_extension("part");
    tokio::fs::write(&destination, b"already downloaded").await.unwrap();
    tokio::fs::write(&resume_artifact, b"partial").await.unwrap();

    let manager = UniversalDownloadManager::new("test-manager".to_string());
    let task = manager
        .file_download_task(
            &"https://example.com/model.bin".to_string(),
            &destination,
            FileCheck::None,
            Some("already downloaded".len() as u64),
        )
        .await
        .unwrap();

    let state = task.state().await;
    assert_eq!(state.phase, FileDownloadPhase::Downloaded);
    assert!(!resume_artifact.exists());
}

#[tokio::test]
async fn test_universal_manager_uses_crc_observation_and_deletes_invalid_file() {
    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join("model.bin");
    tokio::fs::write(&destination, b"corrupt").await.unwrap();

    let manager = UniversalDownloadManager::new("test-manager".to_string());
    let task = manager
        .file_download_task(
            &"https://example.com/model.bin".to_string(),
            &destination,
            FileCheck::CRC("AAAAAA==".to_string()),
            Some("corrupt".len() as u64),
        )
        .await
        .unwrap();

    let state = task.state().await;
    assert_eq!(state.phase, FileDownloadPhase::NotDownloaded);
    assert!(!destination.exists());
}

#[tokio::test]
async fn test_universal_manager_writes_crc_cache_from_action_plan() {
    let bytes = b"valid model";
    let crc_bytes = crc32c::crc32c(bytes).to_be_bytes();
    let expected_crc = base64::engine::general_purpose::STANDARD.encode(crc_bytes);
    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join("model.bin");
    let crc_path = std::path::PathBuf::from(format!("{}.crc", destination.display()));
    tokio::fs::write(&destination, bytes).await.unwrap();

    let manager = UniversalDownloadManager::new("test-manager".to_string());
    let task = manager
        .file_download_task(
            &"https://example.com/model.bin".to_string(),
            &destination,
            FileCheck::CRC(expected_crc.clone()),
            Some(bytes.len() as u64),
        )
        .await
        .unwrap();

    let state = task.state().await;
    assert_eq!(state.phase, FileDownloadPhase::Downloaded);
    assert_eq!(tokio::fs::read_to_string(crc_path).await.unwrap(), expected_crc);
}
