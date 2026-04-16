#![cfg(target_vendor = "apple")]

mod common;

use std::time::Duration;

use common::TestDownloadManager;
use download_manager::{FileCheck, FileDownloadManagerType, FileDownloadPhase};
use tokio::time::sleep;

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_manager_resume_data_deleted_on_resume() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("[TEST] ========== RESUME DATA CLEANUP TEST ==========");

    let test_manager =
        TestDownloadManager::new("test_apple_manager_resume_data_deleted_on_resume", FileDownloadManagerType::Apple)
            .await?;
    let dest_path = test_manager.dest_path("test_file");
    let resume_path = std::path::PathBuf::from(format!("{}.resume_data", dest_path.display()));
    let lock_path = std::path::PathBuf::from(format!("{}.lock", dest_path.display()));

    // Clean up any existing artifacts
    let _ = std::fs::remove_file(&dest_path);
    let _ = std::fs::remove_file(format!("{}.crc", dest_path.display()));
    let _ = std::fs::remove_file(&resume_path);
    let _ = std::fs::remove_file(&lock_path);

    // Create task and start download
    let task = test_manager
        .manager
        .file_download_task(
            &test_manager.test_file.url,
            &dest_path,
            FileCheck::CRC(test_manager.test_file.crc.clone()),
            Some(test_manager.test_file.size),
        )
        .await?;

    task.download().await?;

    // Wait until we are Downloading and have some bytes
    let mut attempts = 0;
    let max_attempts = 200;
    loop {
        let state = task.state().await;
        if matches!(state.phase, FileDownloadPhase::Downloading) && state.total_bytes > 0 {
            break;
        }
        attempts += 1;
        if attempts >= max_attempts {
            panic!("Timeout waiting for Downloading state");
        }
        sleep(Duration::from_millis(50)).await;
    }

    // Pause to create resume_data
    task.pause().await?;

    // Wait for Paused and resume_data to appear
    let mut attempts = 0;
    loop {
        let state = task.state().await;
        let has_resume = resume_path.exists();
        if matches!(state.phase, FileDownloadPhase::Paused) && has_resume {
            break;
        }
        attempts += 1;
        if attempts >= max_attempts {
            panic!(
                "Timeout waiting for Paused state with resume_data present (state={:?}, resume_exists={})",
                state.phase, has_resume
            );
        }
        sleep(Duration::from_millis(50)).await;
    }

    // Resume download
    task.download().await?;

    // After resume begins, resume_data must be deleted promptly
    let mut attempts = 0;
    loop {
        let state = task.state().await;
        let resume_exists = resume_path.exists();
        // Accept either Downloading without resume_data, or very fast completion with no resume_data
        if (!resume_exists)
            && (matches!(state.phase, FileDownloadPhase::Downloading)
                || matches!(state.phase, FileDownloadPhase::Downloaded))
        {
            break;
        }
        attempts += 1;
        if attempts >= max_attempts {
            panic!(
                "Timeout waiting for resume_data deletion on resume (state={:?}, resume_exists={})",
                state.phase, resume_exists
            );
        }
        sleep(Duration::from_millis(50)).await;
    }

    tracing::info!("[TEST] ✓ resume_data removed upon resuming download");

    Ok(())
}
