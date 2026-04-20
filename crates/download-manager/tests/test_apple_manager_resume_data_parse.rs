#![cfg(target_vendor = "apple")]

mod common;

use std::time::Duration;

use common::TestDownloadManager;
use download_manager::{
    FileCheck, FileDownloadManagerType, FileDownloadPhase, managers::apple::URLSessionDownloadTaskResumeData,
};
use tokio::time::{sleep, timeout};
use tokio_stream::StreamExt;

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_manager_resume_data_contains_reasonable_progress_bytes() -> Result<(), Box<dyn std::error::Error>> {
    let test_manager = TestDownloadManager::new(
        "test_apple_manager_resume_data_contains_reasonable_progress_bytes",
        FileDownloadManagerType::Apple,
    )
    .await?;
    let mgr = &test_manager.manager;
    let dest_path = test_manager.dest_path("test_file");

    let task = mgr
        .file_download_task(
            &test_manager.test_file.url,
            &dest_path,
            FileCheck::CRC(test_manager.test_file.crc.clone()),
            Some(test_manager.test_file.size),
        )
        .await?;
    let mut stream = task.progress().await?;

    // Start the download and wait until we have some meaningful progress
    task.download().await?;

    let mut observed_downloaded: Option<u64> = None;
    let mut observed_total: Option<u64> = None;

    let _ = timeout(Duration::from_secs(60), async {
        while let Some(next) = stream.next().await {
            if let Ok(state) = next {
                match state.phase {
                    FileDownloadPhase::Downloading => {
                        observed_total = Some(state.total_bytes);
                        // Choose a modest threshold to ensure resumeData has content
                        let target = if state.total_bytes > 0 {
                            // 2% or at least 5 MiB, whichever is lower
                            let two_percent = state.total_bytes / 50;
                            two_percent.min(5 * 1024 * 1024)
                        } else {
                            5 * 1024 * 1024
                        };
                        if state.downloaded_bytes >= target && target > 0 {
                            observed_downloaded = Some(state.downloaded_bytes);
                            break;
                        }
                    },
                    FileDownloadPhase::Downloaded => {
                        // Completed too fast; take the final number
                        observed_downloaded = Some(state.downloaded_bytes);
                        observed_total = Some(state.total_bytes);
                        break;
                    },
                    _ => {},
                }
            }
        }
    })
    .await;

    // Ensure we recorded some progress
    let observed_downloaded = observed_downloaded.unwrap_or(0);
    assert!(observed_downloaded > 0, "no progress observed before cancellation");

    // Pause to generate resume data
    task.pause().await?;

    // Check if download completed
    let final_state = task.state().await;
    if matches!(final_state.phase, FileDownloadPhase::Downloaded) {
        tracing::info!("Download completed before pause - test passed (no resume data to parse)");
        test_manager.cleanup().await;
        return Ok(());
    }

    // Read the written resume data file
    let resume_path = format!("{}.resume_data", dest_path.display());
    let resume_bytes = std::fs::read(&resume_path)?;
    assert!(!resume_bytes.is_empty(), "resume_data file is empty: {}", resume_path);

    // Parse and verify
    let resume_data = URLSessionDownloadTaskResumeData::from_bytes(&resume_bytes).expect("failed to parse resume_data");

    // Expect positive received bytes
    let bytes_received = resume_data.bytes_received.expect("bytes_received should be present");
    assert!(bytes_received > 0, "parsed bytes_received is zero");

    // Tolerance: allow up to 10% or 10 MiB difference, whichever is larger
    let diff = if bytes_received > observed_downloaded {
        bytes_received - observed_downloaded
    } else {
        observed_downloaded - bytes_received
    };
    let ten_percent = observed_downloaded / 10;
    let max_allowed = ten_percent.max(10 * 1024 * 1024);
    assert!(
        diff <= max_allowed,
        "parsed bytes_received ({}) diverges too much from observed ({})",
        bytes_received,
        observed_downloaded
    );

    // Note: total_bytes is not reliably available in resume data, so we skip that check

    // Small wait to allow any background cleanup
    sleep(Duration::from_millis(200)).await;

    test_manager.cleanup().await;
    Ok(())
}
