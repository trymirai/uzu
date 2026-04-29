use std::{path::PathBuf, sync::Arc};

use download_manager::{
    DownloadError, FileCheck, FileDownloadPhase, FileDownloadTask,
    backends::universal::{UniversalBackend, UniversalBackendContext},
    file_download_task_actor::{GenericFileDownloadTask, ProgressCounters, PublicProjection},
    reducer::InitialLifecycleState,
    traits::DownloadConfig,
};
use uuid::Uuid;

use crate::common::MockRegistry;

async fn task(
    initial_lifecycle_state: InitialLifecycleState
) -> Result<GenericFileDownloadTask<UniversalBackend>, Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    Ok(GenericFileDownloadTask::spawn(
        Arc::new(DownloadConfig {
            download_id: Uuid::nil(),
            source_url: served_file.file.url.clone(),
            destination: PathBuf::from(served_file.file.name.clone()),
            file_check: FileCheck::None,
            expected_bytes: Some(120),
            manager_id: "test-manager".to_string(),
        }),
        Arc::new(UniversalBackendContext::default()),
        initial_lifecycle_state,
        PublicProjection::None,
        ProgressCounters {
            downloaded_bytes: 40,
            total_bytes: 120,
        },
    ))
}

#[tokio::test]
async fn test_generic_file_download_task_state_is_cached_projection() -> Result<(), Box<dyn std::error::Error>> {
    let task = task(InitialLifecycleState::Paused {
        part_path: PathBuf::from("model.bin.part"),
    })
    .await?;

    let state = task.state().await;

    assert_eq!(state.phase, FileDownloadPhase::Paused);
    assert_eq!(state.downloaded_bytes, 40);
    assert_eq!(state.total_bytes, 120);
    Ok(())
}

#[tokio::test]
async fn test_generic_file_download_task_pause_from_downloaded_is_invalid() -> Result<(), Box<dyn std::error::Error>> {
    let task = task(InitialLifecycleState::Downloaded {
        file_path: PathBuf::from("model.bin"),
        crc_path: None,
    })
    .await?;

    let result = task.pause().await;

    assert_eq!(result, Err(DownloadError::InvalidStateTransition));
    Ok(())
}

#[tokio::test]
async fn test_generic_file_download_task_cancel_from_paused_is_idempotent_cleanup(
) -> Result<(), Box<dyn std::error::Error>> {
    let task = task(InitialLifecycleState::Paused {
        part_path: PathBuf::from("model.bin.part"),
    })
    .await?;

    task.cancel().await.unwrap();
    task.cancel().await.unwrap();

    let state = task.state().await;
    assert_eq!(state.phase, FileDownloadPhase::NotDownloaded);
    Ok(())
}
