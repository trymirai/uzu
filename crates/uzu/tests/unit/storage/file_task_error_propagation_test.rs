use std::sync::Arc;

use download_manager::{FileDownloadState, FileDownloadTask};
use mock_registry::MockRegistry;
use rstest::rstest;
use tokio::{runtime::Handle as TokioHandle, sync::broadcast::channel as tokio_broadcast_channel};
use uzu::storage::types::{DownloadPhase, DownloadState, Item};

use crate::common::failing_file_task::{FailingFileTask, StubManager};

#[derive(Clone, Copy, Debug)]
enum FailingOperation {
    Download,
    Pause,
    Cancel,
    Detach,
}

#[rstest]
#[case::download(FailingOperation::Download)]
#[case::pause(FailingOperation::Pause)]
#[case::cancel(FailingOperation::Cancel)]
#[case::detach(FailingOperation::Detach)]
#[tokio::test(flavor = "multi_thread")]
async fn file_task_operation_errors_propagate(
    #[case] operation: FailingOperation
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let total_bytes = served_file.file.size as u64;
    let downloaded_bytes = 1;

    let temporary_directory = tempfile::tempdir()?;
    let cache_path = temporary_directory.path().join("model");
    let initial_file_state = match operation {
        FailingOperation::Download => FileDownloadState::not_downloaded(total_bytes),
        FailingOperation::Pause | FailingOperation::Cancel | FailingOperation::Detach => {
            FileDownloadState::downloading(downloaded_bytes, total_bytes)
        },
    };
    let initial_download_state = match operation {
        FailingOperation::Download => DownloadState::not_downloaded(total_bytes as i64),
        FailingOperation::Pause | FailingOperation::Cancel | FailingOperation::Detach => {
            DownloadState::downloading(downloaded_bytes as i64, total_bytes as i64)
        },
    };
    let file_task = FailingFileTask::new(
        served_file.file.url.clone(),
        cache_path.join(&served_file.file.name),
        initial_file_state,
    );
    let file_task = match operation {
        FailingOperation::Download => file_task.with_download_error("forced download failure"),
        FailingOperation::Pause => file_task.with_pause_error("forced pause failure"),
        FailingOperation::Cancel | FailingOperation::Detach => file_task.with_cancel_error("forced cancel failure"),
    };

    let (storage_broadcast_sender, _) = tokio_broadcast_channel(1);
    let item = Item::new(
        "test-model".to_string(),
        Arc::new(vec![served_file.file.clone()]),
        cache_path,
        initial_download_state,
        Arc::new(StubManager::new()),
        vec![Arc::new(file_task) as Arc<dyn FileDownloadTask>],
        TokioHandle::current(),
        storage_broadcast_sender,
    );

    let operation_result = match operation {
        FailingOperation::Download => item.download().await,
        FailingOperation::Pause => item.pause().await,
        FailingOperation::Cancel => item.cancel().await,
        FailingOperation::Detach => item.detach_active_downloads().await,
    };
    assert!(operation_result.is_err(), "{operation:?} must propagate file task errors");

    if matches!(operation, FailingOperation::Pause) {
        let state = item.state().await;
        assert!(
            !matches!(state.phase, DownloadPhase::Paused {}),
            "Item::pause failed; public phase must not falsely report Paused; got {:?}",
            state.phase,
        );
    }

    Ok(())
}
