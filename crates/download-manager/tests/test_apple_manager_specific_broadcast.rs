mod common;

use std::time::Duration;

use common::TestDownloadManager;
use download_manager::{FileCheck, FileDownloadManagerType, FileDownloadPhase};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_manager_specific_broadcast() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("=== Task-Specific Broadcast Test ===");

    let test_manager =
        TestDownloadManager::new("test_apple_manager_specific_broadcast", FileDownloadManagerType::Apple).await?;
    let dest_path = test_manager.dest_path("test_file");
    let manager = &test_manager.manager;

    tracing::info!("Creating task...");
    let task = manager
        .file_download_task(
            &test_manager.test_file.url,
            &dest_path,
            FileCheck::CRC(test_manager.test_file.crc.clone()),
            Some(test_manager.test_file.size),
        )
        .await?;

    tracing::info!("Subscribing to TASK-SPECIFIC progress...");
    let mut task_progress = task.progress().await?;
    tracing::info!("[TASK_TEST] Subscribed to task-specific progress channel");

    let pb = ProgressBar::new(0);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})",
            )
            .unwrap()
            .progress_chars("█▓▒░ "),
    );

    let pb_clone = pb.clone();
    let (completion_tx, mut completion_rx) = tokio::sync::mpsc::channel::<()>(1);

    let monitor_task = tokio::spawn(async move {
        const IDLE_TIMEOUT: Duration = Duration::from_secs(30);
        let mut update_count = 0;

        loop {
            match tokio::time::timeout(IDLE_TIMEOUT, task_progress.next()).await {
                Ok(Some(Ok(state))) => {
                    update_count += 1;

                    let progress = if state.total_bytes > 0 {
                        state.downloaded_bytes as f64 / state.total_bytes as f64 * 100.0
                    } else {
                        0.0
                    };

                    tracing::info!(
                        "[TASK_TEST] Update #{}: {:?} - {}/{} bytes ({:.1}%)",
                        update_count,
                        state.phase,
                        state.downloaded_bytes,
                        state.total_bytes,
                        progress
                    );

                    if pb_clone.length().unwrap_or(0) == 0 && state.total_bytes > 0 {
                        pb_clone.set_length(state.total_bytes);
                    }
                    pb_clone.set_position(state.downloaded_bytes);

                    match state.phase {
                        FileDownloadPhase::Downloaded => {
                            pb_clone.finish_with_message("✓ Downloaded");
                            let _ = completion_tx.send(()).await;
                            break;
                        },
                        FileDownloadPhase::Error(msg) => {
                            pb_clone.abandon_with_message(format!("Error: {}", msg));
                            break;
                        },
                        _ => {},
                    }
                },
                Ok(Some(Err(e))) => {
                    tracing::info!("Stream error: {}", e);
                },
                Ok(None) => {
                    tracing::info!("Stream ended after {} updates", update_count);
                    break;
                },
                Err(_) => {
                    tracing::info!("No updates for 30s after {} updates", update_count);
                    pb_clone.abandon_with_message("Timeout");
                    break;
                },
            }
        }

        update_count
    });

    tracing::info!("Starting download...");
    task.download().await?;
    tracing::info!("[TASK_TEST] Download started, task_id={}", task.download_id());

    let completed = tokio::time::timeout(Duration::from_secs(60), completion_rx.recv()).await.is_ok();

    let update_count = monitor_task.await?;

    tracing::info!("\nResults:");
    tracing::info!("   Task-specific updates received: {}", update_count);
    tracing::info!("   Download completed: {}", completed);

    if update_count > 0 {
        tracing::info!("   ✓ TASK-SPECIFIC broadcast works!");
        tracing::info!("   This proves URLSession delegates ARE firing");
    } else {
        tracing::info!("   ❌ No task-specific updates");
        tracing::info!("   URLSession delegates not firing in test environment");
    }

    test_manager.cleanup().await;
    Ok(())
}
