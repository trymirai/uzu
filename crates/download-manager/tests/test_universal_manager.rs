use std::{path::Path, sync::Arc};

use download_manager::{
    FileCheck, FileDownloadManager, FileDownloadPhase, backends::universal::UniversalDownloadManager,
};

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
