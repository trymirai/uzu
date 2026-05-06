use std::sync::Arc;

use download_manager::{FileDownloadState, FileDownloadTask};
use mock_registry::MockRegistry;
use tokio::{runtime::Handle as TokioHandle, sync::broadcast::channel as tokio_broadcast_channel};
use uzu::storage::types::{DownloadState, Item};

use crate::common::failing_file_task::{FailingFileTask, StubManager};

#[tokio::test(flavor = "multi_thread")]
async fn download_propagates_file_task_error() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let total_bytes = served_file.file.size as u64;

    let temp_dir = tempfile::tempdir()?;
    let cache_path = temp_dir.path().join("model");
    let file_task = Arc::new(
        FailingFileTask::new(
            served_file.file.url.clone(),
            cache_path.join(&served_file.file.name),
            FileDownloadState::not_downloaded(total_bytes),
        )
        .with_download_error("forced download failure"),
    );

    let (storage_broadcast_sender, _) = tokio_broadcast_channel(1);
    let item = Item::new(
        "test-model".to_string(),
        Arc::new(vec![served_file.file.clone()]),
        cache_path,
        DownloadState::not_downloaded(total_bytes as i64),
        Arc::new(StubManager::new()),
        vec![file_task as Arc<dyn FileDownloadTask>],
        TokioHandle::current(),
        storage_broadcast_sender,
    );

    let download_result = item.download().await;
    assert!(download_result.is_err(), "Item::download must propagate file_task download errors");

    Ok(())
}
