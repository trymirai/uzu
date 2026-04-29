use std::path::PathBuf;

use download_manager::{
    FileCheck, FileDownloadPhase,
    file_download_task_actor::{ProgressCounters, PublicProjection, project_public_state},
    reducer::InitialLifecycleState,
    traits::DownloadConfig,
};
use uuid::Uuid;

use crate::common::MockRegistry;

async fn config(expected_bytes: Option<u64>) -> Result<DownloadConfig, Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    Ok(DownloadConfig {
        download_id: Uuid::nil(),
        source_url: served_file.file.url.clone(),
        destination: PathBuf::from(served_file.file.name.clone()),
        file_check: FileCheck::None,
        expected_bytes,
        manager_id: "test-manager".to_string(),
    })
}

#[tokio::test]
async fn test_project_public_state_sticky_error_overrides_lifecycle() -> Result<(), Box<dyn std::error::Error>> {
    let config = config(Some(100)).await?;
    let state = project_public_state(
        &InitialLifecycleState::Downloaded {
            file_path: PathBuf::from("model.bin"),
            crc_path: None,
        },
        &PublicProjection::StickyError("boom".to_string()),
        ProgressCounters {
            downloaded_bytes: 100,
            total_bytes: 100,
        },
        &config,
    );

    assert_eq!(state.phase, FileDownloadPhase::Error("boom".to_string()));
    Ok(())
}

#[tokio::test]
async fn test_project_public_state_locked_by_other_overrides_lifecycle() -> Result<(), Box<dyn std::error::Error>> {
    let config = config(Some(100)).await?;
    let state = project_public_state(
        &InitialLifecycleState::NotDownloaded,
        &PublicProjection::LockedByOther("other-manager".to_string()),
        ProgressCounters::default(),
        &config,
    );

    assert_eq!(state.phase, FileDownloadPhase::LockedByOther("other-manager".to_string()));
    Ok(())
}

#[tokio::test]
async fn test_project_public_state_paused_uses_expected_bytes_when_total_unknown() -> Result<(), Box<dyn std::error::Error>>
{
    let config = config(Some(120)).await?;
    let state = project_public_state(
        &InitialLifecycleState::Paused {
            part_path: PathBuf::from("model.bin.part"),
        },
        &PublicProjection::None,
        ProgressCounters {
            downloaded_bytes: 40,
            total_bytes: 0,
        },
        &config,
    );

    assert_eq!(state.downloaded_bytes, 40);
    assert_eq!(state.total_bytes, 120);
    assert_eq!(state.phase, FileDownloadPhase::Paused);
    Ok(())
}

#[tokio::test]
async fn test_project_public_state_downloaded_prefers_expected_bytes() -> Result<(), Box<dyn std::error::Error>> {
    let config = config(Some(120)).await?;
    let state = project_public_state(
        &InitialLifecycleState::Downloaded {
            file_path: PathBuf::from("model.bin"),
            crc_path: None,
        },
        &PublicProjection::None,
        ProgressCounters {
            downloaded_bytes: 100,
            total_bytes: 100,
        },
        &config,
    );

    assert_eq!(state.downloaded_bytes, 120);
    assert_eq!(state.total_bytes, 120);
    assert_eq!(state.phase, FileDownloadPhase::Downloaded);
    Ok(())
}
