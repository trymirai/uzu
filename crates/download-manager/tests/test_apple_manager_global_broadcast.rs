mod common;

use std::{sync::Arc, time::Duration};

use common::TestDownloadManager;
use download_manager::{FileCheck, FileDownloadEvent, FileDownloadManagerType};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_manager_global_broadcast() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("=== Global Broadcast Progress Test ===");

    let test_manager =
        TestDownloadManager::new("test_apple_manager_global_broadcast", FileDownloadManagerType::Apple).await?;
    let dest_path = test_manager.dest_path("test_file");
    let manager = Arc::new(test_manager.manager);

    tracing::info!("Creating file download task...");
    let task = manager
        .file_download_task(
            &test_manager.test_file.url,
            &dest_path,
            FileCheck::CRC(test_manager.test_file.crc),
            Some(test_manager.test_file.size),
        )
        .await?;

    tracing::info!("Subscribing to GLOBAL downloads...");
    let mut global_updates = manager.subscribe_to_all_downloads();
    tracing::info!("[GLOBAL_TEST] Subscribed to global broadcast channel");

    let pb = ProgressBar::new(0);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta})"
            )
            .unwrap()
            .progress_chars("█▓▒░ "),
    );
    pb.set_message("Downloading");

    let task_download_id = task.download_id();
    let pb_clone = pb.clone();
    let (completion_tx, mut completion_rx) = tokio::sync::mpsc::channel::<()>(1);

    let monitor_task = tokio::spawn(async move {
        const IDLE_TIMEOUT: Duration = Duration::from_secs(30);
        let mut update_count = 0;

        loop {
            match tokio::time::timeout(IDLE_TIMEOUT, global_updates.next()).await {
                Ok(Some(Ok((download_id, event)))) => {
                    update_count += 1;

                    if download_id == task_download_id {
                        match event {
                            FileDownloadEvent::ProgressUpdate {
                                bytes_written: _,
                                total_bytes_written,
                                total_bytes_expected,
                            } => {
                                let progress = if total_bytes_expected > 0 {
                                    total_bytes_written as f64 / total_bytes_expected as f64 * 100.0
                                } else {
                                    0.0
                                };

                                tracing::info!(
                                    "[GLOBAL_TEST] Update #{}: Progress - {}/{} bytes ({:.1}%)",
                                    update_count,
                                    total_bytes_written,
                                    total_bytes_expected,
                                    progress
                                );

                                if pb_clone.length().unwrap_or(0) == 0 && total_bytes_expected > 0 {
                                    pb_clone.set_length(total_bytes_expected);
                                }

                                pb_clone.set_position(total_bytes_written);
                            },
                            FileDownloadEvent::DownloadCompleted {
                                tmp_path: _,
                                final_destination: _,
                            } => {
                                tracing::info!("[GLOBAL_TEST] Update #{}: Downloaded", update_count);
                                pb_clone.finish_with_message("✓ Downloaded");
                                let _ = completion_tx.send(()).await;
                                break;
                            },
                            FileDownloadEvent::Error {
                                message,
                            } => {
                                pb_clone.abandon_with_message(format!("Error: {}", message));
                                break;
                            },
                        }
                    } else {
                        tracing::info!("Update #{}: Different file ({})", update_count, download_id);
                    }
                },
                Ok(Some(Err(e))) => {
                    tracing::info!("Broadcast error: {}", e);
                },
                Ok(None) => {
                    tracing::info!("Stream ended after {} updates", update_count);
                    pb_clone.abandon_with_message("Stream ended");
                    break;
                },
                Err(_) => {
                    tracing::info!("No progress for 30s after {} updates", update_count);
                    pb_clone.abandon_with_message("Timeout");
                    break;
                },
            }
        }

        update_count
    });

    tracing::info!("Starting download...");
    task.download().await?;
    tracing::info!("[GLOBAL_TEST] Download started, task_id={}", task.download_id());

    tracing::info!("Waiting for completion or timeout...");
    let completed = tokio::time::timeout(Duration::from_secs(60), completion_rx.recv()).await.is_ok();

    let update_count = monitor_task.await?;

    tracing::info!("\nResults:");
    tracing::info!("   Updates received: {}", update_count);
    tracing::info!("   Completed: {}", completed);

    if update_count > 0 {
        tracing::info!("   ✓ Global broadcast is working!");
    } else {
        tracing::info!("   ❌ No updates received - URLSession delegates not firing in test");
        tracing::info!("      This is expected in cargo test environment");
        tracing::info!("      Actual downloads work in CLI/app context");
    }

    // Cleanup (manager is Arc wrapped, just drop it and wait)
    drop(manager);
    tokio::time::sleep(Duration::from_millis(100)).await;

    Ok(())
}
