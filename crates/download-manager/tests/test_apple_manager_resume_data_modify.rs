#![cfg(target_vendor = "apple")]

mod common;

use std::time::Duration;

use common::TestDownloadManager;
use download_manager::{
    FileCheck, FileDownloadManagerType, FileDownloadPhase, managers::apple::URLSessionDownloadTaskResumeData,
};
use tokio::time::timeout;
use tokio_stream::StreamExt;

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_manager_resume_data_can_be_modified_and_saved() -> Result<(), Box<dyn std::error::Error>> {
    let test_manager = TestDownloadManager::new(
        "test_apple_manager_resume_data_can_be_modified_and_saved",
        FileDownloadManagerType::Apple,
    )
    .await?;
    let mgr = &test_manager.manager;
    let dest_path = test_manager.dest_path("test_file");

    tracing::info!("Cleaning up any existing artifacts...");
    let _ = std::fs::remove_file(&dest_path);
    let _ = std::fs::remove_file(format!("{}.crc", dest_path.display()));
    let _ = std::fs::remove_file(format!("{}.resume_data", dest_path.display()));

    let task = mgr
        .file_download_task(
            &test_manager.test_file.url,
            &dest_path,
            FileCheck::CRC(test_manager.test_file.crc.clone()),
            Some(test_manager.test_file.size),
        )
        .await?;

    let initial_state = task.state().await;
    tracing::info!("Initial state: {:?}", initial_state.phase);

    // Start download BEFORE subscribing to stream
    task.download().await?;
    tracing::info!("Download started...");

    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut stream = task.progress().await?;
    tracing::info!("Subscribed to progress stream, waiting for updates...");

    let mut observed_downloaded: Option<u64> = None;
    let mut observed_total: Option<u64> = None;

    let _ = timeout(Duration::from_secs(60), async {
        let mut update_count = 0;
        while let Some(next) = stream.next().await {
            if let Ok(state) = next {
                update_count += 1;
                if update_count % 10 == 1 {
                    tracing::info!(
                        "Update #{}: {:?} - {}/{} bytes",
                        update_count,
                        state.phase,
                        state.downloaded_bytes,
                        state.total_bytes
                    );
                }

                match state.phase {
                    FileDownloadPhase::Downloading => {
                        if state.total_bytes > 0 {
                            observed_total = Some(state.total_bytes);
                            let target = state.total_bytes / 50; // 2%
                            if state.downloaded_bytes >= target && target > 0 {
                                observed_downloaded = Some(state.downloaded_bytes);
                                tracing::info!("Reached target at {} bytes", state.downloaded_bytes);
                                break;
                            }
                        }
                    },
                    FileDownloadPhase::Downloaded => {
                        tracing::info!("Download completed: {} bytes", state.downloaded_bytes);
                        observed_downloaded = Some(state.downloaded_bytes);
                        observed_total = Some(state.total_bytes);
                        break;
                    },
                    _ => {},
                }
            }
        }
        tracing::info!("Stream ended after {} updates", update_count);
    })
    .await;

    assert!(observed_downloaded.is_some(), "should have observed progress");
    assert!(observed_total.is_some(), "should have observed total bytes");

    // Pause to generate resume data
    task.pause().await?;

    // Check if download completed
    let final_state = task.state().await;
    if matches!(final_state.phase, FileDownloadPhase::Downloaded) {
        tracing::info!("Download completed before pause - test passed (no resume data to modify)");
        test_manager.cleanup().await;
        return Ok(());
    }

    // Read the resume data file
    let resume_path = format!("{}.resume_data", dest_path.display());
    let original_bytes = std::fs::read(&resume_path)?;

    // Parse resume data
    let mut resume_data = URLSessionDownloadTaskResumeData::from_bytes(&original_bytes)?;

    tracing::info!(
        "Original resume data: bytes_received={:?}, bytes_expected={:?}",
        resume_data.bytes_received,
        resume_data.bytes_expected_to_receive
    );

    // Verify bytes_received is present
    assert!(resume_data.bytes_received.is_some(), "bytes_received should be present");

    // Modify: add expected bytes if not present
    if resume_data.bytes_expected_to_receive.is_none() {
        resume_data.bytes_expected_to_receive = observed_total;
        tracing::info!("Injected expected bytes: {:?}", resume_data.bytes_expected_to_receive);

        // Save modified resume data
        resume_data.save_to_file(&resume_path)?;

        // Re-read and verify the modification was saved
        let saved_bytes = std::fs::read(&resume_path)?;
        let reloaded = URLSessionDownloadTaskResumeData::from_bytes(&saved_bytes)?;

        tracing::info!(
            "Reloaded resume data: bytes_received={:?}, bytes_expected={:?}",
            reloaded.bytes_received,
            reloaded.bytes_expected_to_receive
        );

        assert_eq!(reloaded.bytes_received, resume_data.bytes_received, "bytes_received should be preserved");
        assert_eq!(reloaded.bytes_expected_to_receive, observed_total, "bytes_expected_to_receive should be saved");

        tracing::info!("✓ Successfully modified and saved resume data");
    }

    test_manager.cleanup().await;
    Ok(())
}
