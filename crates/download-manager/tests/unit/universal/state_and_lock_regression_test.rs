use std::sync::Arc;

use download_manager::{DownloadError, FileCheck, FileDownloadManagerType, FileDownloadPhase, create_download_manager};
use tokio::runtime::Handle as TokioHandle;

use crate::common::{mock_download_server::RouteBehavior, scenarios::DownloadTestContext};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_foreign_lock_makes_download_fail() {
    let context = DownloadTestContext::new(
        "model.safetensors",
        RouteBehavior::StallAt {
            byte_offset: 64 * 1024,
        },
    )
    .await;
    let manager = create_download_manager(
        FileDownloadManagerType::Universal,
        Some("universal-lock-owner".to_string()),
        TokioHandle::current(),
    )
    .await
    .expect("failed to create download manager");
    let task = manager
        .file_download_task(&context.payload.file.url, &context.destination, context.file_check(), context.file_size())
        .await
        .expect("failed to create file download task");

    task.download().await.expect("failed to start owner download");
    context.wait_for_bytes(64 * 1024).await;
    assert!(context.lock_path().exists(), "download should own a lock while running");

    let other_manager = create_download_manager(
        FileDownloadManagerType::Universal,
        Some("universal-lock-other".to_string()),
        TokioHandle::current(),
    )
    .await
    .expect("failed to create second manager");
    let other_task = other_manager
        .file_download_task(
            &context.payload.file.url,
            &context.destination,
            FileCheck::CRC(context.payload.crc32c()),
            context.file_size(),
        )
        .await
        .expect("failed to create second task");

    let other_state = other_task.state().await;
    assert!(matches!(other_state.phase, FileDownloadPhase::LockedByOther(_)));
    assert!(matches!(other_task.download().await, Err(DownloadError::LockedByOther(_))));

    task.cancel().await.expect("failed to cancel owner download");
    context.release_stall().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_concurrent_task_creation_returns_same_task() {
    let context = DownloadTestContext::new("tokenizer.json", RouteBehavior::Normal).await;
    let manager = create_download_manager(
        FileDownloadManagerType::Universal,
        Some("universal-single-flight".to_string()),
        TokioHandle::current(),
    )
    .await
    .expect("failed to create download manager");

    let (first, second) = tokio::join!(
        manager.file_download_task(&context.payload.file.url, &context.destination, context.file_check(), context.file_size()),
        manager.file_download_task(&context.payload.file.url, &context.destination, context.file_check(), context.file_size()),
    );
    let first = first.expect("first task should be created");
    let second = second.expect("second task should be created");

    assert!(Arc::ptr_eq(&first, &second), "same URL and destination should return the same task object");
}
