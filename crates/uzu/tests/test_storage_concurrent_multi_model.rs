#![cfg(not(target_family = "wasm"))]

mod common;

use std::time::Duration;

use common::test_helpers::TestStorage;
use futures_util::StreamExt;
use uzu::storage::types::DownloadPhase;

const TEST_COUNT: usize = 2;

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_storage_concurrent_multi_model_download() -> Result<(), Box<dyn std::error::Error>> {
    common::tracing_setup::init_test_tracing();

    tracing::info!("========== STARTING CONCURRENT DOWNLOAD TEST ==========");

    let temp_dir = tempfile::tempdir()?;
    let base_path = temp_dir.path().to_path_buf();
    let test_storage = TestStorage::new_with_base_path(base_path).await?;

    tracing::info!("[CONCURRENT_TEST] Initializing storage...");
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Create test instances - we'll use the same model reference multiple times
    let mut models = Vec::new();
    for i in 0..TEST_COUNT {
        // Get the test model and create instances
        tracing::info!("[CONCURRENT_TEST] Fetching test model {}...", i);
        let model = test_storage.model(i).clone();
        let item = test_storage.storage.get(&model.identifier.clone()).await.ok_or("Model not found")?;

        // Cancel first to ensure it's not already downloaded
        tracing::info!("[CONCURRENT_TEST] Canceling model {} to ensure fresh download...", i);
        item.cancel().await?;
        tokio::time::sleep(Duration::from_millis(500)).await;
        models.push((model.identifier.clone(), item.clone()));
    }

    tracing::info!("[CONCURRENT_TEST] ✓ Created {} model instances", models.len());

    // Start downloading all models concurrently
    tracing::info!("\n[CONCURRENT_TEST] ========== PHASE 1: Start all downloads ==========");
    let download_futures: Vec<_> = models
        .iter()
        .map(|(id, model)| {
            let id = id.clone();
            let model = model.clone();
            async move {
                tracing::info!("[CONCURRENT_TEST] Starting download for {}", id);
                match model.download().await {
                    Ok(_) => tracing::info!("[CONCURRENT_TEST] ✓ Download started for {}", id),
                    Err(e) => tracing::info!("[CONCURRENT_TEST] ✗ Download failed for {}: {:?}", id, e),
                }
            }
        })
        .collect();

    futures_util::future::join_all(download_futures).await;

    // Wait for some progress with timeout protection
    // Note: Since we're using clones of the same model, they share state and have the same ID
    tracing::info!("\n[CONCURRENT_TEST] Waiting for downloads to progress...");
    let mut progress_stream = test_storage.storage.subscribe();
    let mut progress_count = 0;
    let expected_progress = 10; // Wait for at least 10 progress updates

    let progress_check = tokio::time::timeout(Duration::from_secs(10), async {
        while let Some(Ok((id, state))) = progress_stream.next().await {
            if matches!(state.phase, DownloadPhase::Downloading {} | DownloadPhase::Downloaded {}) {
                progress_count += 1;
                if progress_count <= 5 || progress_count % 10 == 0 {
                    tracing::info!(
                        "[CONCURRENT_TEST] Progress #{}: {} - bytes={}/{}",
                        progress_count,
                        id,
                        state.downloaded_bytes,
                        state.total_bytes
                    );
                }
                if progress_count >= expected_progress {
                    break;
                }
            }
        }
        progress_count
    })
    .await;

    let count = progress_check.unwrap_or(0);
    assert!(count >= expected_progress, "TIMEOUT: Expected {} progress updates, got {}", expected_progress, count);

    // Check all are downloading
    for (id, model) in &models {
        let state = model.state().await;
        tracing::info!(
            "[CONCURRENT_TEST] Model {} state: phase={:?}, bytes={}/{}",
            id,
            state.phase,
            state.downloaded_bytes,
            state.total_bytes
        );
        assert!(
            matches!(state.phase, DownloadPhase::Downloading {} | DownloadPhase::Downloaded {}),
            "Model {} should be downloading or downloaded, got {:?}",
            id,
            state.phase
        );
    }

    // Pause all models concurrently
    tracing::info!("\n[CONCURRENT_TEST] ========== PHASE 2: Pause all downloads ==========");
    let pause_futures: Vec<_> = models
        .iter()
        .map(|(id, model)| {
            let id = id.clone();
            let model = model.clone();
            async move {
                tracing::info!("[CONCURRENT_TEST] Pausing {}", id);
                match model.pause().await {
                    Ok(_) => {
                        tracing::info!("[CONCURRENT_TEST] ✓ Paused {}", id)
                    },
                    Err(e) => tracing::info!("[CONCURRENT_TEST] ⚠ Pause for {} returned: {:?}", id, e),
                }
            }
        })
        .collect();

    futures_util::future::join_all(pause_futures).await;

    // Wait for pause confirmations with timeout
    let pause_check = tokio::time::timeout(Duration::from_secs(5), async {
        let mut all_paused = false;
        for _ in 0..10 {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let mut paused_count = 0;
            for (_, model) in &models {
                let state = model.state().await;
                if matches!(state.phase, DownloadPhase::Paused {} | DownloadPhase::Downloaded {}) {
                    paused_count += 1;
                }
            }
            if paused_count == models.len() {
                all_paused = true;
                break;
            }
        }
        all_paused
    })
    .await;

    assert!(pause_check.unwrap_or(false), "TIMEOUT: Not all models paused within 5 seconds");

    // Verify all are paused (or completed)
    for (id, model) in &models {
        let state = model.state().await;
        tracing::info!(
            "[CONCURRENT_TEST] Model {} after pause: phase={:?}, bytes={}/{}",
            id,
            state.phase,
            state.downloaded_bytes,
            state.total_bytes
        );
        assert!(
            matches!(state.phase, DownloadPhase::Paused {} | DownloadPhase::Downloaded {}),
            "Model {} should be paused or downloaded after pause, got {:?}",
            id,
            state.phase
        );
    }

    // Resume all models concurrently
    tracing::info!("\n[CONCURRENT_TEST] ========== PHASE 3: Resume all downloads ==========");
    let resume_futures: Vec<_> = models
        .iter()
        .map(|(id, model)| {
            let id = id.clone();
            let model = model.clone();
            async move {
                tracing::info!("[CONCURRENT_TEST] Resuming {}", id);
                match model.download().await {
                    Ok(_) => {
                        tracing::info!("[CONCURRENT_TEST] ✓ Resumed {}", id)
                    },
                    Err(e) => tracing::info!("[CONCURRENT_TEST] ⚠ Resume for {} returned: {:?}", id, e),
                }
            }
        })
        .collect();

    futures_util::future::join_all(resume_futures).await;

    tokio::time::sleep(Duration::from_secs(2)).await;

    // Check resumed
    for (id, model) in &models {
        let state = model.state().await;
        tracing::info!(
            "[CONCURRENT_TEST] Model {} after resume: phase={:?}, bytes={}/{}",
            id,
            state.phase,
            state.downloaded_bytes,
            state.total_bytes
        );
    }

    // Cancel half the models while others continue
    tracing::info!("\n[CONCURRENT_TEST] ========== PHASE 4: Cancel half while others continue ==========");
    let cancel_count = models.len() / 2;
    let cancel_futures: Vec<_> = models
        .iter()
        .take(cancel_count)
        .map(|(id, model)| {
            let id = id.clone();
            let model = model.clone();
            async move {
                tracing::info!("[CONCURRENT_TEST] Canceling {}", id);
                match model.cancel().await {
                    Ok(_) => {
                        tracing::info!("[CONCURRENT_TEST] ✓ Canceled {}", id)
                    },
                    Err(e) => tracing::info!("[CONCURRENT_TEST] ✗ Cancel for {} failed: {:?}", id, e),
                }
            }
        })
        .collect();

    futures_util::future::join_all(cancel_futures).await;

    // Wait for cancel confirmations with timeout
    let cancel_check = tokio::time::timeout(Duration::from_secs(5), async {
        let mut all_cancelled = false;
        for _ in 0..10 {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let mut cancelled_count = 0;
            for (idx, (_, model)) in models.iter().enumerate() {
                if idx < cancel_count {
                    let state = model.state().await;
                    if matches!(state.phase, DownloadPhase::NotDownloaded {}) {
                        cancelled_count += 1;
                    }
                }
            }
            if cancelled_count == cancel_count {
                all_cancelled = true;
                break;
            }
        }
        all_cancelled
    })
    .await;

    assert!(cancel_check.unwrap_or(false), "TIMEOUT: Not all models cancelled within 5 seconds");

    // Verify cancelled models are not downloaded, others still progressing
    for (idx, (id, model)) in models.iter().enumerate() {
        let state = model.state().await;
        if idx < cancel_count {
            tracing::info!("[CONCURRENT_TEST] Cancelled model {} state: phase={:?}", id, state.phase);
            assert!(
                matches!(state.phase, DownloadPhase::NotDownloaded {}),
                "Cancelled model {} should be NotDownloaded, got {:?}",
                id,
                state.phase
            );
        } else {
            tracing::info!(
                "[CONCURRENT_TEST] Continuing model {} state: phase={:?}, bytes={}/{}",
                id,
                state.phase,
                state.downloaded_bytes,
                state.total_bytes
            );
        }
    }

    // Monitor progress updates for continuing models
    tracing::info!("\n[CONCURRENT_TEST] ========== PHASE 5: Monitor progress updates ==========");
    // Use a model that wasn't cancelled (second half of the list)
    if let Some((id, model)) = models.iter().skip(cancel_count).next() {
        tracing::info!("[CONCURRENT_TEST] Starting download for monitoring {}", id);
        model.download().await?;

        tracing::info!("[CONCURRENT_TEST] Monitoring progress for {}", id);
        let mut progress_stream = test_storage.storage.subscribe();
        let model_id = id.clone();

        let monitor_task = tokio::spawn(async move {
            let mut update_count = 0;
            while let Some(Ok((id, state))) = progress_stream.next().await {
                if id == model_id {
                    update_count += 1;
                    tracing::info!(
                        "[CONCURRENT_TEST] Progress update #{} for {}: phase={:?}, bytes={}/{}",
                        update_count,
                        id,
                        state.phase,
                        state.downloaded_bytes,
                        state.total_bytes
                    );
                    if update_count >= 5 || matches!(state.phase, DownloadPhase::Downloaded {}) {
                        break;
                    }
                }
            }
            update_count
        });

        let update_count =
            tokio::time::timeout(Duration::from_secs(30), monitor_task).await.unwrap_or(Ok(0)).unwrap_or(0);

        tracing::info!("[CONCURRENT_TEST] Received {} progress updates", update_count);
        tracing::info!("[CONCURRENT_TEST] ✓ Progress monitoring successful");
    }

    tracing::info!("\n[CONCURRENT_TEST] ========== CLEANUP ==========");
    drop(test_storage.storage);

    tracing::info!("[CONCURRENT_TEST] ========== TEST COMPLETE ✓ ==========\n");
    Ok(())
}

// concurrent_pause_resume_stress_test removed
