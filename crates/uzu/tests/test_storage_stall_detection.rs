#![cfg(not(target_family = "wasm"))]

mod common;

use std::time::Duration;

use common::test_helpers::TestStorage;
use futures_util::StreamExt;
use uzu::storage::types::DownloadPhase;

#[tokio::test(flavor = "multi_thread")]
async fn test_stall_detection_no_progress() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("\n[STALL_TEST] ========== NO PROGRESS STALL DETECTION TEST ==========");

    let temp_dir = tempfile::tempdir()?;
    let base_path = temp_dir.path().to_path_buf();
    let test_storage = TestStorage::new_with_base_path(base_path).await?;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
    tracing::info!("[STALL_TEST] Starting download...");
    model.download().await?;

    // Monitor for progress with stall detection
    let mut progress_stream = test_storage.storage.subscribe();
    let mut last_progress_time = std::time::Instant::now();
    let mut last_bytes = 0u64;
    let max_stall_duration = Duration::from_secs(10);
    let check_interval = Duration::from_millis(500);

    tracing::info!("[STALL_TEST] Monitoring for stalls (max stall: {:?})...", max_stall_duration);

    let stall_check = tokio::time::timeout(Duration::from_secs(60), async {
        let mut check_count = 0;
        loop {
            tokio::select! {
                Some(Ok((id, state))) = progress_stream.next() => {
                    if id == test_storage.model(0).identifier() {
                        let current_bytes = state.downloaded_bytes;
                        if current_bytes > last_bytes as i64 {
                            let elapsed = last_progress_time.elapsed();
                            tracing::info!(
                                "[STALL_TEST] Progress: {} bytes (delta: {}, time_since_last: {:?})",
                                current_bytes,
                                current_bytes - last_bytes as i64,
                                elapsed
                            );
                            last_bytes = current_bytes as u64;
                            last_progress_time = std::time::Instant::now();
                        }

                        if matches!(state.phase, DownloadPhase::Downloaded {}) {
                            tracing::info!("[STALL_TEST] ✓ Download completed");
                            return Ok::<_, String>(());
                        }
                    }
                }
                _ = tokio::time::sleep(check_interval) => {
                    check_count += 1;
                    let elapsed = last_progress_time.elapsed();

                    if elapsed > max_stall_duration {
                        return Err(format!(
                            "STALL DETECTED: No progress for {:?} (last bytes: {})",
                            elapsed, last_bytes
                        ));
                    }

                    if check_count % 10 == 0 {
                        tracing::info!(
                            "[STALL_TEST] Check #{}: No stall detected (last progress: {:?} ago)",
                            check_count, elapsed
                        );
                    }

                    // Stop after reaching 10% for this test
                    let state = model.state().await;
                    if state.total_bytes > 0 && state.downloaded_bytes > state.total_bytes / 10 {
                        tracing::info!("[STALL_TEST] ✓ Reached 10%, stopping test");
                        return Ok(());
                    }
                }
            }
        }
    })
    .await;

    match stall_check {
        Ok(Ok(())) => {
            tracing::info!("[STALL_TEST] ✓ No stalls detected");
        },
        Ok(Err(stall_msg)) => {
            panic!("{}", stall_msg);
        },
        Err(_) => {
            panic!("TIMEOUT: Test took longer than 60 seconds");
        },
    }

    // Cancel the download
    tracing::info!("[STALL_TEST] Canceling download...");
    model.cancel().await?;

    tracing::info!("\n[STALL_TEST] ========== CLEANUP ==========");
    drop(test_storage.storage);

    tracing::info!("[STALL_TEST] ========== TEST COMPLETE ✓ ==========\n");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_stall_detection_broadcast_liveness() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("\n[BROADCAST_TEST] ========== BROADCAST LIVENESS TEST ==========");

    let temp_dir = tempfile::tempdir()?;
    let base_path = temp_dir.path().to_path_buf();
    let test_storage = TestStorage::new_with_base_path(base_path).await?;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
    tracing::info!("[BROADCAST_TEST] Starting download...");
    model.download().await?;

    // Verify we receive regular broadcast updates
    let mut progress_stream = test_storage.storage.subscribe();
    let mut update_count = 0;
    let mut last_update_time = std::time::Instant::now();
    let max_gap = Duration::from_secs(5);

    tracing::info!("[BROADCAST_TEST] Monitoring broadcast updates (max gap: {:?})...", max_gap);

    let broadcast_check = tokio::time::timeout(Duration::from_secs(30), async {
        while let Some(Ok((id, state))) = progress_stream.next().await {
            if id == test_storage.model(0).identifier() {
                let gap = last_update_time.elapsed();
                update_count += 1;

                if gap > max_gap {
                    return Err(format!(
                        "BROADCAST GAP TOO LARGE: {:?} between updates #{} and #{}",
                        gap,
                        update_count - 1,
                        update_count
                    ));
                }

                if update_count <= 10 || update_count % 50 == 0 {
                    tracing::info!(
                        "[BROADCAST_TEST] Update #{}: bytes={}/{}, gap={:?}",
                        update_count,
                        state.downloaded_bytes,
                        state.total_bytes,
                        gap
                    );
                }

                last_update_time = std::time::Instant::now();

                // Stop after 100 updates or completion
                if update_count >= 100 || matches!(state.phase, DownloadPhase::Downloaded {}) {
                    return Ok::<_, String>(update_count);
                }
            }
        }
        Err("Stream ended unexpectedly".to_string())
    })
    .await;

    match broadcast_check {
        Ok(Ok(count)) => {
            tracing::info!("[BROADCAST_TEST] ✓ Received {} updates, all gaps < {:?}", count, max_gap);
            assert!(count >= 10, "Should receive at least 10 updates");
        },
        Ok(Err(gap_msg)) => {
            panic!("{}", gap_msg);
        },
        Err(_) => {
            panic!("TIMEOUT: Did not receive enough updates within 30 seconds");
        },
    }

    // Cancel the download
    tracing::info!("[BROADCAST_TEST] Canceling download...");
    model.cancel().await?;

    tracing::info!("\n[BROADCAST_TEST] ========== CLEANUP ==========");
    drop(test_storage.storage);

    tracing::info!("[BROADCAST_TEST] ========== TEST COMPLETE ✓ ==========\n");
    Ok(())
}
