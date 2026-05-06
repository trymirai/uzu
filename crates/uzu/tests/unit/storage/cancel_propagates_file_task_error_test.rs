use std::sync::Arc;

use download_manager::{FileDownloadState, FileDownloadTask};
use mock_registry::MockRegistry;
use tokio::{runtime::Handle as TokioHandle, sync::broadcast::channel as tokio_broadcast_channel};
use uzu::storage::types::{DownloadState, Item};

use crate::common::failing_file_task::{FailingFileTask, StubManager};

#[tokio::test(flavor = "multi_thread")]
async fn cancel_propagates_file_task_error() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let total_bytes = served_file.file.size as u64;
    let downloaded_bytes = 1;

    let temp_dir = tempfile::tempdir()?;
    let cache_path = temp_dir.path().join("model");
    let file_task = Arc::new(
        FailingFileTask::new(
            served_file.file.url.clone(),
            cache_path.join(&served_file.file.name),
            FileDownloadState::downloading(downloaded_bytes, total_bytes),
        )
        .with_cancel_error("forced cancel failure"),
    );

    let (storage_broadcast_sender, _) = tokio_broadcast_channel(1);
    let item = Item::new(
        "test-model".to_string(),
        Arc::new(vec![served_file.file.clone()]),
        cache_path,
        DownloadState::downloading(downloaded_bytes as i64, total_bytes as i64),
        Arc::new(StubManager::new()),
        vec![file_task as Arc<dyn FileDownloadTask>],
        TokioHandle::current(),
        storage_broadcast_sender,
    );

    let cancel_result = item.cancel().await;
    assert!(cancel_result.is_err(), "Item::cancel must propagate file_task cancel errors");

    Ok(())
}
