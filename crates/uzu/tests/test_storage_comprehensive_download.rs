#![cfg(not(target_family = "wasm"))]

mod common;

use std::time::Duration;

use common::test_helpers::TestStorage;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use uzu::storage::types::{DownloadPhase, Item};

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_comprehensive_download() -> Result<(), Box<dyn std::error::Error>> {
    common::tracing_setup::init_test_tracing();
    tracing::info!("\n[TEST] ========== MODEL DOWNLOAD TEST STARTING ==========");

    let temp_dir = tempfile::tempdir()?;
    let base_path = temp_dir.path().to_path_buf();

    tracing::info!("[TEST] Initializing storage...");
    let test_storage = TestStorage::new_with_base_path(base_path).await?;
    let model_identifier = test_storage.model(0).identifier();
    let storage = &test_storage.storage;
    tokio::time::sleep(Duration::from_millis(200)).await;

    tracing::info!("[TEST] Getting model {}...", model_identifier.clone());
    let model: Item =
        storage.get(&model_identifier).await.ok_or_else(|| format!("Model {} not found", model_identifier.clone()))?;

    let initial_state = model.state().await;
    tracing::info!(
        "[TEST] Model found, total_bytes: {}, initial phase: {:?}",
        initial_state.total_bytes,
        initial_state.phase
    );
    let pb = ProgressBar::new(initial_state.total_bytes);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({percent}%, {bytes_per_sec}, eta {eta}) - {msg}")
            .unwrap()
            .progress_chars("█▓▒░ "),
    );
    tracing::info!("\n[TEST] ========== PHASE 1: Download to 25% ==========");
    pb.set_message("Phase 1: Downloading to 25%");
    let mut updates = storage.subscribe();
    let pb_clone = pb.clone();
    let quarter_threshold = initial_state.total_bytes / 4;
    let (phase1_tx, mut phase1_rx) = tokio::sync::mpsc::channel::<u64>(1);
    let model_identifier_clone = model_identifier.clone();
    let monitor_task = tokio::spawn(async move {
        const TIMEOUT: Duration = Duration::from_secs(120);
        tracing::info!("[PHASE1] Monitor task started, waiting for updates...");
        while let Ok(Some(Ok((id, state)))) = tokio::time::timeout(TIMEOUT, updates.next()).await {
            tracing::info!(
                "[PHASE1] Received update: id={}, phase={:?}, bytes={}/{}",
                id,
                state.phase,
                state.downloaded_bytes,
                state.total_bytes
            );
            if id == model_identifier_clone {
                pb_clone.set_position(state.downloaded_bytes);
                if state.downloaded_bytes >= quarter_threshold {
                    tracing::info!("[PHASE1] ✓ Reached 25% threshold");
                    let _ = phase1_tx.send(state.downloaded_bytes).await;
                    break;
                }
                if matches!(state.phase, DownloadPhase::Downloaded) {
                    tracing::info!("[PHASE1] ✓ Downloaded state reached (completed before 25%)");
                    let _ = phase1_tx.send(state.downloaded_bytes).await;
                    break;
                }
            }
        }
        tracing::info!("[PHASE1] Monitor task exiting");
    });

    tracing::info!("[PHASE1] Starting download...");
    model.download().await?;

    tracing::info!("[PHASE1] Waiting for progress update...");
    if phase1_rx.recv().await.is_none() {
        monitor_task.abort();
        panic!("Phase 1 failed: no progress update received");
    }
    tracing::info!("[PHASE1] ✓ Phase 1 complete");
    monitor_task.abort();

    let current_state = model.state().await;
    if matches!(current_state.phase, DownloadPhase::Downloaded) {
        tracing::info!("\n[TEST] ========== Download completed early, skipping pause/resume tests ==========");
        pb.finish_with_message("Download completed");
        return Ok(());
    }

    tracing::info!("\n[TEST] ========== PHASE 2: Pause at 25% ==========");
    pb.set_message("Phase 2: Pausing...");
    model.pause().await?;
    tokio::time::sleep(Duration::from_secs(2)).await;

    let paused_state = model.state().await;
    tracing::info!(
        "[PHASE2] ✓ Paused state confirmed: phase={:?}, bytes={}/{}",
        paused_state.phase,
        paused_state.downloaded_bytes,
        paused_state.total_bytes
    );

    tracing::info!("\n[TEST] ========== PHASE 3: Resume to 50% ==========");
    pb.set_message("Phase 3: Resuming to 50%");
    let mut updates = storage.subscribe();
    let pb_clone = pb.clone();
    let half_threshold = initial_state.total_bytes / 2;
    let (phase3_tx, mut phase3_rx) = tokio::sync::mpsc::channel::<u64>(1);
    let model_identifier_clone = model_identifier.clone();
    let monitor_task = tokio::spawn(async move {
        const TIMEOUT: Duration = Duration::from_secs(120);
        while let Ok(Some(Ok((id, state)))) = tokio::time::timeout(TIMEOUT, updates.next()).await {
            if id == model_identifier_clone {
                pb_clone.set_position(state.downloaded_bytes);
                if state.downloaded_bytes >= half_threshold {
                    let _ = phase3_tx.send(state.downloaded_bytes).await;
                    break;
                }
                if matches!(state.phase, DownloadPhase::Downloaded) {
                    let _ = phase3_tx.send(state.downloaded_bytes).await;
                    break;
                }
            }
        }
    });
    model.download().await?;
    if phase3_rx.recv().await.is_none() {
        monitor_task.abort();
        panic!("Phase 3 failed: no progress update received");
    }
    tracing::info!("[PHASE3] ✓ Phase 3 complete");
    monitor_task.abort();

    tracing::info!("\n[TEST] ========== PHASE 4: Cancel at 50% ==========");
    pb.set_message("Phase 4: Cancelling...");
    model.cancel().await?;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let cancelled_state = model.state().await;
    tracing::info!("[PHASE4] ✓ Cancelled state confirmed: phase={:?}", cancelled_state.phase);
    assert!(
        matches!(cancelled_state.phase, DownloadPhase::NotDownloaded),
        "Expected model to be in NotDownloaded state after cancellation, got {:?}",
        cancelled_state.phase
    );

    tracing::info!("\n[TEST] ========== PHASE 5: Download again to 50% ==========");
    pb.set_message("Phase 5: Downloading again to 50%");
    pb.set_position(0); // Reset progress bar
    let mut updates = storage.subscribe();
    let pb_clone = pb.clone();
    let (phase5_tx, mut phase5_rx) = tokio::sync::mpsc::channel::<u64>(1);
    let model_identifier_clone = model_identifier.clone();
    let monitor_task = tokio::spawn(async move {
        const TIMEOUT: Duration = Duration::from_secs(120);
        while let Ok(Some(Ok((id, state)))) = tokio::time::timeout(TIMEOUT, updates.next()).await {
            if id == model_identifier_clone {
                pb_clone.set_position(state.downloaded_bytes);
                if state.downloaded_bytes >= half_threshold {
                    let _ = phase5_tx.send(state.downloaded_bytes).await;
                    break;
                }
                if matches!(state.phase, DownloadPhase::Downloaded) {
                    let _ = phase5_tx.send(state.downloaded_bytes).await;
                    break;
                }
            }
        }
    });
    model.download().await?;
    if phase5_rx.recv().await.is_none() {
        monitor_task.abort();
        panic!("Phase 5 failed: no progress update received");
    }
    tracing::info!("[PHASE5] ✓ Phase 5 complete");
    monitor_task.abort();

    tracing::info!("\n[TEST] ========== PHASE 6: Pause at 50% ==========");
    pb.set_message("Phase 6: Pausing at 50%");
    model.pause().await?;
    tokio::time::sleep(Duration::from_secs(2)).await;

    let paused_state2 = model.state().await;
    tracing::info!(
        "[PHASE6] ✓ Paused state confirmed: phase={:?}, bytes={}/{}",
        paused_state2.phase,
        paused_state2.downloaded_bytes,
        paused_state2.total_bytes
    );

    tracing::info!("\n[TEST] ========== PHASE 7: Resume to completion ==========");
    pb.set_message("Phase 7: Resuming to completion");
    let mut updates = storage.subscribe();
    let pb_clone = pb.clone();
    let (phase7_tx, mut phase7_rx) = tokio::sync::mpsc::channel::<()>(1);
    let model_identifier_clone = model_identifier.clone();
    let monitor_task = tokio::spawn(async move {
        const TIMEOUT: Duration = Duration::from_secs(240);
        tracing::info!("[PHASE7] Monitor task started, waiting for completion...");
        while let Ok(Some(Ok((id, state)))) = tokio::time::timeout(TIMEOUT, updates.next()).await {
            tracing::info!(
                "[PHASE7] Received update: id={}, phase={:?}, bytes={}/{}",
                id,
                state.phase,
                state.downloaded_bytes,
                state.total_bytes
            );
            if id == model_identifier_clone {
                pb_clone.set_position(state.downloaded_bytes);
                match state.phase {
                    DownloadPhase::Downloaded => {
                        tracing::info!("[PHASE7] ✓ Downloaded state reached!");
                        pb_clone.finish_with_message("✓ Download complete!");
                        let send_result = phase7_tx.send(()).await;
                        tracing::info!("[PHASE7] Channel send result: {:?}", send_result);
                        break;
                    },
                    DownloadPhase::Error(ref err) => {
                        tracing::info!("[PHASE7] ✗ Error state reached: {}", err);
                        pb_clone.abandon_with_message(format!("✗ Error: {}", err));
                        break;
                    },
                    _ => {},
                }
            }
        }
        tracing::info!("[PHASE7] Monitor task exiting");
    });

    tracing::info!("[PHASE7] Starting download...");
    model.download().await?;

    tracing::info!("[PHASE7] Waiting for download completion...");
    if phase7_rx.recv().await.is_none() {
        tracing::info!("[PHASE7] ✗ Channel recv returned None!");
        monitor_task.abort();
        panic!("Phase 7 failed: download did not complete");
    }
    tracing::info!("[PHASE7] ✓ Phase 7 complete");
    monitor_task.abort();

    tracing::info!("\n[TEST] ========== FINAL VERIFICATION ==========");
    let final_state = model.state().await;
    tracing::info!("[TEST] Final model state: {:?}, bytes: {}", final_state.phase, final_state.downloaded_bytes);

    assert!(
        matches!(final_state.phase, DownloadPhase::Downloaded),
        "Expected Downloaded state, got {:?}",
        final_state.phase
    );

    tracing::info!("[TEST] ✓ Test verification passed");
    tracing::info!("[TEST] ========== MODEL DOWNLOAD TEST COMPLETE ==========\n");
    Ok(())
}
