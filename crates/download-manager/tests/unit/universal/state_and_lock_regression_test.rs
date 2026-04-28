use std::sync::Arc;

use chrono::Utc;
use download_manager::{DownloadError, FileCheck, FileDownloadManagerType, FileDownloadPhase, create_download_manager};
use serde_json::to_vec;
use tokio::fs::write as tokio_write;
use tokio::runtime::Handle as TokioHandle;

use crate::common::{mock_download_server::RouteBehavior, scenarios::DownloadTestContext};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_foreign_lock_makes_download_fail() {
    let context = DownloadTestContext::new("model.safetensors", RouteBehavior::Normal).await;
    let foreign_lock = serde_json::json!({
        "manager_id": "other-manager",
        "acquired_at": Utc::now(),
        "process_id": std::process::id(),
    });
    tokio_write(context.lock_path(), to_vec(&foreign_lock).unwrap()).await.unwrap();

    let manager = create_download_manager(FileDownloadManagerType::Universal, None, TokioHandle::current())
    .await
    .unwrap();
    let task = manager
        .file_download_task(
            &context.payload.file.url,
            &context.destination,
            FileCheck::CRC(context.payload.crc32c()),
            context.file_size(),
        )
        .await
        .unwrap();

    let state = task.state().await;
    assert!(matches!(state.phase, FileDownloadPhase::LockedByOther(_)));
    assert!(matches!(task.download().await, Err(DownloadError::LockedByOther(_))));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_concurrent_task_creation_returns_same_task() {
    let context = DownloadTestContext::new("tokenizer.json", RouteBehavior::Normal).await;
    let manager = create_download_manager(FileDownloadManagerType::Universal, None, TokioHandle::current()).await.unwrap();

    let (first, second) = tokio::join!(
        manager.file_download_task(&context.payload.file.url, &context.destination, context.file_check(), context.file_size()),
        manager.file_download_task(&context.payload.file.url, &context.destination, context.file_check(), context.file_size()),
    );
    let first = first.unwrap();
    let second = second.unwrap();

    assert!(Arc::ptr_eq(&first, &second), "same URL and destination should return the same task object");
}
