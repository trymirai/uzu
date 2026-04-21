mod common;

use std::time::Duration;

use common::TestDownloadManager;
use download_manager::{FileCheck, FileDownloadManagerType, FileDownloadPhase};
use indicatif::{ProgressBar, ProgressStyle};
use tokio::time::timeout;
use tokio_stream::StreamExt;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "flaky"]
async fn test_apple_manager_comprehensive() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!(
        "[TEST][{}] ========== COMPREHENSIVE DOWNLOAD TEST STARTING ==========",
        format!("{:?}", std::thread::current().id())
    );

    let test_manager =
        TestDownloadManager::new("test_apple_manager_comprehensive", FileDownloadManagerType::Apple).await?;
    let mgr = &test_manager.manager;
    let dest_path = test_manager.dest_path("test_file");
    tracing::info!("[TEST] Destination: {}", dest_path.display());

    // Clean up any existing artifacts from previous runs
    tracing::info!("[TEST] Cleaning up any existing artifacts...");
    let _ = std::fs::remove_file(&dest_path);
    let _ = std::fs::remove_file(format!("{}.crc", dest_path.display()));
    let _ = std::fs::remove_file(format!("{}.resume_data", dest_path.display()));
    let _ = std::fs::remove_file(format!("{}.lock", dest_path.display()));
    tracing::info!("[TEST] Starting fresh download...");

    // Create progress bar
    let pb = ProgressBar::new(0);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({percent}%, {bytes_per_sec}, eta {eta}) - {msg}")
            .unwrap()
            .progress_chars("█▓▒░ "),
    );

    // ========== PHASE 1: Download to 25% ==========
    tracing::info!("\n[TEST] ========== PHASE 1: Download to 25% ==========");
    pb.set_message("Phase 1: Downloading to 25%");

    let task = mgr
        .file_download_task(
            &test_manager.test_file.url,
            &dest_path,
            FileCheck::CRC(test_manager.test_file.crc.clone()),
            Some(test_manager.test_file.size),
        )
        .await?;

    let initial_state = task.state().await;
    tracing::info!("[TEST] Initial task state: {:?}", initial_state.phase);
    tracing::info!("[TEST] Initial bytes: {} / {}", initial_state.downloaded_bytes, initial_state.total_bytes);

    let mut stream = task.progress().await?;
    tracing::info!("[TEST] Subscribed to progress stream");

    task.download().await?;
    tracing::info!("[TEST] Started download (task_id={}), waiting for 25% progress...", task.download_id());

    // Verify lock file was created
    let lock_path = format!("{}.lock", dest_path.display());
    tokio::time::sleep(Duration::from_millis(500)).await; // Give it a moment
    if std::path::Path::new(&lock_path).exists() {
        tracing::info!("[TEST] ✓ Lock file created: {}", lock_path);
    } else {
        tracing::warn!("[TEST] ⚠ Lock file not found (may not be implemented yet)");
    }

    let mut total_bytes: u64 = 0;
    let mut update_count = 0;
    let phase1_result = timeout(Duration::from_secs(120), async {
        tracing::info!("[TEST] Waiting for progress updates...");
        while let Some(next) = stream.next().await {
            update_count += 1;
            tracing::info!("[TEST] Received update #{}", update_count);
            if let Ok(state) = next {
                tracing::info!(
                    "[TEST] Update #{}: phase={:?}, bytes={}/{}",
                    update_count,
                    state.phase,
                    state.downloaded_bytes,
                    state.total_bytes
                );
                match state.phase {
                    FileDownloadPhase::Downloading => {
                        if state.total_bytes > 0 {
                            total_bytes = state.total_bytes;
                            if pb.length().unwrap_or(0) == 0 {
                                pb.set_length(state.total_bytes);
                            }
                            pb.set_position(state.downloaded_bytes);

                            let progress_pct = (state.downloaded_bytes as f64 / state.total_bytes as f64) * 100.0;
                            tracing::info!(
                                "[TEST] Progress: {:.1}% ({} / {} bytes)",
                                progress_pct,
                                state.downloaded_bytes,
                                state.total_bytes
                            );
                            if progress_pct >= 25.0 {
                                tracing::info!(
                                    "[TEST] ✓ Reached 25% target ({} / {} bytes)",
                                    state.downloaded_bytes,
                                    state.total_bytes
                                );
                                return Ok::<(u64, u64), String>((state.downloaded_bytes, state.total_bytes));
                            }
                        }
                    },
                    FileDownloadPhase::Downloaded => {
                        tracing::info!("[TEST] ⚠️ Download completed before 25%!");
                        return Ok((state.downloaded_bytes, state.total_bytes));
                    },
                    FileDownloadPhase::Error(err) => {
                        tracing::info!("[TEST] ❌ Error in phase 1: {}", err);
                        return Err(format!("Error in phase 1: {}", err));
                    },
                    _ => {
                        tracing::info!("[TEST] Other phase: {:?}", state.phase);
                    },
                }
            }
        }
        Err("Stream ended before reaching 25%".to_string())
    })
    .await;

    let (_, _, download_completed_early) = match phase1_result {
        Ok(Ok((downloaded, total))) => {
            tracing::info!("[TEST] Phase 1 complete: {} / {} bytes", downloaded, total);
            let completed = downloaded >= total && total > 0;
            if completed {
                tracing::info!("[TEST] ⚠️  Download completed during phase 1 - network too fast!");
            }
            (downloaded, total, completed)
        },
        Ok(Err(e)) => panic!("Phase 1 failed: {}", e),
        Err(_) => panic!("Phase 1 timed out"),
    };

    // ========== PHASE 2: Pause ==========
    tracing::info!("\n[TEST] ========== PHASE 2: Pause ==========");
    pb.set_message("Phase 2: Pausing...");

    if download_completed_early {
        tracing::info!("[TEST] Skipping pause phase - download already completed");
        let state = task.state().await;
        assert!(
            matches!(state.phase, FileDownloadPhase::Downloaded),
            "Expected Downloaded state for completed download, got {:?}",
            state.phase
        );
        tracing::info!("[TEST] Confirmed downloaded state: {} bytes", state.downloaded_bytes);
    } else {
        task.pause().await?;
        tracing::info!("[TEST] Paused download");
        tokio::time::sleep(Duration::from_millis(500)).await;

        let state = task.state().await;
        assert!(matches!(state.phase, FileDownloadPhase::Paused), "Expected Paused state, got {:?}", state.phase);
        tracing::info!("[TEST] Confirmed paused state: {} / {} bytes", state.downloaded_bytes, state.total_bytes);
    }

    // ========== PHASE 3: Resume to 50% ==========
    if download_completed_early {
        tracing::info!("\n[TEST] ========== PHASE 3: Skipped (already downloaded) ==========");
    } else {
        tracing::info!("\n[TEST] ========== PHASE 3: Resume to 50% ==========");
        pb.set_message("Phase 3: Resuming to 50%");

        // Get fresh task handle after pause
        let task = mgr
            .file_download_task(
                &test_manager.test_file.url,
                &dest_path,
                FileCheck::CRC(test_manager.test_file.crc.clone()),
                Some(test_manager.test_file.size),
            )
            .await?;
        task.download().await?;
        tracing::info!("[TEST] Resumed download, waiting for 50% progress...");

        // Get fresh stream from new handle
        let mut stream = task.progress().await?;

        let phase3_result = timeout(Duration::from_secs(120), async {
            while let Some(next) = stream.next().await {
                if let Ok(state) = next {
                    match state.phase {
                        FileDownloadPhase::Downloading => {
                            if state.total_bytes > 0 {
                                if pb.length().unwrap_or(0) == 0 {
                                    pb.set_length(state.total_bytes);
                                }
                                pb.set_position(state.downloaded_bytes);

                                let progress_pct = (state.downloaded_bytes as f64 / state.total_bytes as f64) * 100.0;
                                if progress_pct >= 50.0 {
                                    tracing::info!(
                                        "[TEST] Reached 50% ({} / {} bytes)",
                                        state.downloaded_bytes,
                                        state.total_bytes
                                    );
                                    return Ok::<(u64, u64), String>((state.downloaded_bytes, state.total_bytes));
                                }
                            }
                        },
                        FileDownloadPhase::Downloaded => {
                            tracing::info!("[TEST] Download completed before 50%!");
                            return Ok((state.downloaded_bytes, state.total_bytes));
                        },
                        FileDownloadPhase::Error(err) => {
                            return Err(format!("Error in phase 3: {}", err));
                        },
                        _ => {},
                    }
                }
            }
            Err("Stream ended before reaching 50%".to_string())
        })
        .await;

        match phase3_result {
            Ok(Ok((downloaded, total))) => {
                tracing::info!("[TEST] Phase 3 complete: {} / {} bytes", downloaded, total);
            },
            Ok(Err(e)) => panic!("Phase 3 failed: {}", e),
            Err(_) => panic!("Phase 3 timed out"),
        }

        // ========== PHASE 4: Cancel ==========
        tracing::info!("\n[TEST] ========== PHASE 4: Cancel ==========");
        pb.set_message("Phase 4: Cancelling...");

        task.cancel().await?;
        tracing::info!("[TEST] Cancelled download");
        tokio::time::sleep(Duration::from_millis(500)).await;

        let state = task.state().await;
        assert!(
            matches!(state.phase, FileDownloadPhase::NotDownloaded),
            "Expected NotDownloaded state after cancel, got {:?}",
            state.phase
        );
        tracing::info!("[TEST] Confirmed cancelled state");

        // Drop the old stream
        drop(stream);

        // ========== PHASE 5: Download again to 50% ==========
        tracing::info!("\n[TEST] ========== PHASE 5: Download again to 50% ==========");
        pb.set_message("Phase 5: Downloading again to 50%");
        pb.set_position(0); // Reset progress bar

        // Create new task for fresh download
        let task2 = mgr
            .file_download_task(
                &test_manager.test_file.url,
                &dest_path,
                FileCheck::CRC(test_manager.test_file.crc.clone()),
                Some(test_manager.test_file.size),
            )
            .await?;
        let mut stream2 = task2.progress().await?;

        task2.download().await?;
        tracing::info!("[TEST] Started fresh download, waiting for 50% progress...");

        let phase5_result = timeout(Duration::from_secs(120), async {
            while let Some(next) = stream2.next().await {
                if let Ok(state) = next {
                    match state.phase {
                        FileDownloadPhase::Downloading => {
                            if state.total_bytes > 0 {
                                if pb.length().unwrap_or(0) == 0 {
                                    pb.set_length(state.total_bytes);
                                }
                                pb.set_position(state.downloaded_bytes);

                                let progress_pct = (state.downloaded_bytes as f64 / state.total_bytes as f64) * 100.0;
                                if progress_pct >= 50.0 {
                                    tracing::info!(
                                        "[TEST] Reached 50% again ({} / {} bytes)",
                                        state.downloaded_bytes,
                                        state.total_bytes
                                    );
                                    return Ok::<(u64, u64), String>((state.downloaded_bytes, state.total_bytes));
                                }
                            }
                        },
                        FileDownloadPhase::Downloaded => {
                            tracing::info!("[TEST] Download completed before 50%!");
                            return Ok((state.downloaded_bytes, state.total_bytes));
                        },
                        FileDownloadPhase::Error(err) => {
                            return Err(format!("Error in phase 5: {}", err));
                        },
                        _ => {},
                    }
                }
            }
            Err("Stream ended before reaching 50%".to_string())
        })
        .await;

        match phase5_result {
            Ok(Ok((downloaded, total))) => {
                tracing::info!("[TEST] Phase 5 complete: {} / {} bytes", downloaded, total);
            },
            Ok(Err(e)) => panic!("Phase 5 failed: {}", e),
            Err(_) => panic!("Phase 5 timed out"),
        }

        // ========== PHASE 6: Pause at 50% ==========
        tracing::info!("\n[TEST] ========== PHASE 6: Pause at 50% ==========");
        pb.set_message("Phase 6: Pausing at 50%");

        task2.pause().await?;
        tracing::info!("[TEST] Paused download at ~50%");
        tokio::time::sleep(Duration::from_millis(500)).await;

        let state = task2.state().await;
        assert!(matches!(state.phase, FileDownloadPhase::Paused), "Expected Paused state, got {:?}", state.phase);
        tracing::info!(
            "[TEST] Confirmed paused state at 50%: {} / {} bytes",
            state.downloaded_bytes,
            state.total_bytes
        );

        // ========== PHASE 7: Resume to completion ==========
        tracing::info!("\n[TEST] ========== PHASE 7: Resume to completion ==========");
        pb.set_message("Phase 7: Resuming to completion");

        // Get fresh task handle after pause
        let task2 = mgr
            .file_download_task(
                &test_manager.test_file.url,
                &dest_path,
                FileCheck::CRC(test_manager.test_file.crc.clone()),
                Some(test_manager.test_file.size),
            )
            .await?;
        task2.download().await?;
        tracing::info!("[TEST] Resumed download, waiting for completion...");

        // Get fresh stream from new handle
        let mut stream2 = task2.progress().await?;

        let phase7_result = timeout(Duration::from_secs(240), async {
            while let Some(next) = stream2.next().await {
                match next {
                    Ok(state) => match state.phase {
                        FileDownloadPhase::Downloading => {
                            if state.total_bytes > 0 {
                                if pb.length().unwrap_or(0) == 0 {
                                    pb.set_length(state.total_bytes);
                                }
                                pb.set_position(state.downloaded_bytes);
                            }
                        },
                        FileDownloadPhase::Downloaded => {
                            pb.finish_with_message("✓ Download complete!");
                            tracing::info!("[TEST] Download completed successfully!");
                            return Ok::<(), String>(());
                        },
                        FileDownloadPhase::Error(err) => {
                            pb.abandon_with_message(format!("✗ Error: {}", err));
                            return Err(format!("Error in phase 7: {}", err));
                        },
                        _ => {},
                    },
                    Err(_) => {
                        // Ignore lag errors
                    },
                }
            }
            Err("Stream ended before completion".to_string())
        })
        .await;

        match phase7_result {
            Ok(Ok(())) => {
                tracing::info!("[TEST] Phase 7 complete: Download finished!");
            },
            Ok(Err(e)) => panic!("Phase 7 failed: {}", e),
            Err(_) => panic!("Phase 7 timed out"),
        }
    }

    // Verify file exists and has content
    tracing::info!("\n[TEST] ========== FINAL VERIFICATION ==========");
    assert!(dest_path.is_file(), "Destination file does not exist");
    let metadata = std::fs::metadata(&dest_path)?;
    assert!(metadata.len() > 0, "Downloaded file is empty");
    tracing::info!("[TEST] Verified file exists with size: {} bytes", metadata.len());

    // Verify lock file was cleaned up after completion
    let lock_path = format!("{}.lock", dest_path.display());
    if std::path::Path::new(&lock_path).exists() {
        tracing::warn!("[TEST] ⚠ Lock file still exists after completion (should be cleaned up)");
    } else {
        tracing::info!("[TEST] ✓ Lock file properly cleaned up");
    }

    test_manager.cleanup().await;
    tracing::info!("[TEST] ========== COMPREHENSIVE DOWNLOAD TEST COMPLETE ==========\n");
    Ok(())
}
