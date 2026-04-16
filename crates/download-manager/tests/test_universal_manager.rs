mod common;

use common::TestDownloadManager;
use download_manager::{FileCheck, FileDownloadManagerType};

#[tokio::test]
async fn test_universal_manager_create() {
    let test_manager = TestDownloadManager::new("test_universal_manager_create", FileDownloadManagerType::Universal)
        .await
        .expect("Failed to create manager");

    let tasks = test_manager.manager.get_all_file_tasks().await.expect("Failed to get tasks");
    assert_eq!(tasks.len(), 0, "New manager should have no tasks");
}

#[tokio::test]
async fn test_universal_manager_create_download_task() {
    let test_manager =
        TestDownloadManager::new("test_universal_manager_create_download_task", FileDownloadManagerType::Universal)
            .await
            .expect("Failed to create manager");

    let destination = test_manager.dest_path("test_file");

    let task = test_manager
        .manager
        .file_download_task(&test_manager.test_file.url, &destination, FileCheck::None, None)
        .await
        .expect("Failed to create task");

    assert_eq!(task.source_url(), &test_manager.test_file.url);
    assert_eq!(task.destination(), &destination);
}

#[tokio::test]
async fn test_universal_manager_task_state_transitions() {
    let test_manager =
        TestDownloadManager::new("test_universal_manager_task_state_transitions", FileDownloadManagerType::Universal)
            .await
            .expect("Failed to create manager");

    let destination = test_manager.dest_path("test_file");

    let task = test_manager
        .manager
        .file_download_task(
            &test_manager.test_file.url,
            &destination,
            FileCheck::None,
            Some(test_manager.test_file.size),
        )
        .await
        .expect("Failed to create task");

    let initial_state = task.state().await;
    assert!(
        matches!(initial_state.phase, download_manager::FileDownloadPhase::NotDownloaded),
        "Initial state should be NotDownloaded"
    );
}

#[tokio::test]
async fn test_universal_manager_part_file_state_reconciliation() {
    let test_manager = TestDownloadManager::new(
        "test_universal_manager_part_file_state_reconciliation",
        FileDownloadManagerType::Universal,
    )
    .await
    .expect("Failed to create manager");

    let destination = test_manager.dest_path("test_file");

    let part_path = destination.with_extension("part");
    std::fs::write(&part_path, b"partial content").expect("Failed to write part file");

    let task = test_manager
        .manager
        .file_download_task(
            &test_manager.test_file.url,
            &destination,
            FileCheck::None,
            Some(test_manager.test_file.size),
        )
        .await
        .expect("Failed to create task");

    let state = task.state().await;
    assert!(
        matches!(state.phase, download_manager::FileDownloadPhase::Paused),
        "State should be Paused when part file exists, got {:?}",
        state.phase
    );
    assert!(state.downloaded_bytes > 0, "Should have downloaded bytes");
}
