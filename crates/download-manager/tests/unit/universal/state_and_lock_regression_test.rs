use std::{path::PathBuf, sync::Arc};

use chrono::Utc;
use download_manager::{FileCheck, FileDownloadManagerType, FileDownloadPhase, create_download_manager};
use tokio::{fs::write as tokio_write, runtime::Handle as TokioHandle};

use crate::common::MockRegistry;

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_foreign_lock_surfaces_locked_state() {
    let registry = MockRegistry::start().await;
    let model_weights = registry.file("model.safetensors");
    let temp_dir = tempfile::tempdir().unwrap();
    let destination = temp_dir.path().join(&model_weights.file.name);
    let lock_path = PathBuf::from(format!("{}.lock", destination.display()));
    let foreign_lock = serde_json::json!({
        "manager_id": "other-manager",
        "acquired_at": Utc::now(),
        "process_id": std::process::id(),
    });
    tokio_write(&lock_path, serde_json::to_vec(&foreign_lock).unwrap())
        .await
        .unwrap();

    let manager = create_download_manager(FileDownloadManagerType::Universal, TokioHandle::current()).await.unwrap();
    let task = manager
        .file_download_task(
            &model_weights.file.url,
            &destination,
            FileCheck::CRC(model_weights.crc32c()),
            Some(model_weights.file.size as u64),
        )
        .await
        .unwrap();

    let state = task.state().await;
    assert!(matches!(state.phase, FileDownloadPhase::LockedByOther(_)));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_concurrent_task_creation_returns_same_task() {
    let registry = MockRegistry::start().await;
    let tokenizer = registry.file("tokenizer.json");
    let temp_dir = tempfile::tempdir().unwrap();
    let destination = temp_dir.path().join(&tokenizer.file.name);
    let manager = create_download_manager(FileDownloadManagerType::Universal, TokioHandle::current()).await.unwrap();

    let (first, second) = tokio::join!(
        manager.file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()),
            Some(tokenizer.file.size as u64),
        ),
        manager.file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()),
            Some(tokenizer.file.size as u64),
        ),
    );
    let first = first.unwrap();
    let second = second.unwrap();

    assert!(
        Arc::ptr_eq(&first, &second),
        "same URL and destination should return the same task object"
    );
}
