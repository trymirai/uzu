#![cfg(target_vendor = "apple")]

mod common;

use std::time::Duration;

use common::test_helpers::TestStorage;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use uzu::storage::types::DownloadPhase;

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_comprehensive_state_fresh_download_to_completion() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let base_path = temp_dir.path().to_path_buf();

    tracing::info!("\n[SCENARIO_1] Fresh download to completion");
    let test_storage = TestStorage::new_with_base_path(base_path).await?;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
    let state = model.state().await;
    assert!(matches!(state.phase, DownloadPhase::NotDownloaded {}), "Fresh model should be NotDownloaded");

    // Set up progress bar
    let pb = ProgressBar::new(state.total_bytes as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({percent}%, {bytes_per_sec}) - {msg}")
            .unwrap()
            .progress_chars("█▓▒░ "),
    );
    pb.set_message("Downloading to completion");

    // Subscribe to updates
    let mut updates = test_storage.storage.subscribe();
    let pb_clone = pb.clone();
    let (done_tx, mut done_rx) = tokio::sync::mpsc::channel::<()>(1);

    let monitor = tokio::spawn(async move {
        while let Ok(Some(Ok((id, state)))) = tokio::time::timeout(Duration::from_secs(120), updates.next()).await {
            if id == test_storage.model(0).identifier() {
                pb_clone.set_position(state.downloaded_bytes as u64);
                if matches!(state.phase, DownloadPhase::Downloaded {}) {
                    pb_clone.finish_with_message("✓ Downloaded");
                    let _ = done_tx.send(()).await;
                    break;
                }
            }
        }
    });

    model.download().await?;
    tokio::time::sleep(Duration::from_millis(300)).await;

    let state = model.state().await;
    assert!(matches!(state.phase, DownloadPhase::Downloading {}), "Should be Downloading after start");

    // Wait for completion with timeout
    let completed = tokio::time::timeout(Duration::from_secs(600), done_rx.recv()).await;
    monitor.abort();

    assert!(completed.is_ok(), "Download did not complete within timeout");

    let state = model.state().await;
    assert!(matches!(state.phase, DownloadPhase::Downloaded {}), "Should be Downloaded after completion");

    Ok(())
}

#[ignore]
#[tokio::test(flavor = "multi_thread")]
async fn test_storage_comprehensive_state_pause_and_resume() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let base_path = temp_dir.path().to_path_buf();

    let test_storage = TestStorage::new_with_base_path(base_path).await?;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;

    model.download().await?;
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;

    model.pause().await?;
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
    let state = model.state().await;
    assert!(matches!(state.phase, DownloadPhase::Paused {}), "Should be Paused after pause command");
    let paused_progress = state.progress();
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;

    model.download().await?;
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
    let state = model.state().await;
    assert!(matches!(state.phase, DownloadPhase::Downloading {}), "Should be Downloading after resume");
    assert!(state.progress() >= paused_progress, "Progress should not decrease after resume");
    Ok(())
}

#[ignore]
#[tokio::test(flavor = "multi_thread")]
async fn test_storage_comprehensive_state_pause_quit_relaunch_resume() -> Result<(), Box<dyn std::error::Error>> {
    {
        let temp_dir = tempfile::tempdir()?;
        let base_path = temp_dir.path().to_path_buf();

        let test_storage = TestStorage::new_with_base_path(base_path.clone()).await?;
        let storage = &test_storage.storage;
        let model = storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
        model.download().await?;
        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
        let model = storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
        model.pause().await?;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let model = storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
        let state = model.state().await;
        let paused_progress = state.progress();
        let model_dir = test_storage.config.clone().cache_model_path(&test_storage.model(0)).unwrap();
        let mut file_list = Vec::new();
        if model_dir.exists() {
            let mut entries = tokio::fs::read_dir(&model_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let file_name = entry.file_name();
                let name_str = file_name.to_string_lossy().to_string();
                if !name_str.starts_with('.') {
                    let metadata = entry.metadata().await?;
                    file_list.push((name_str.clone(), metadata.len()));
                }
            }
        }
        drop(test_storage);

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let complete_files: Vec<_> =
            file_list.iter().filter(|(name, _)| !name.ends_with(".resume_data")).cloned().collect();

        let test_storage_2 = TestStorage::new_with_base_path(base_path).await?;
        let storage_2 = &test_storage_2.storage;
        let model =
            storage_2.get(&test_storage_2.model(0).identifier()).await.ok_or("Model not found after relaunch")?;
        let state = model.state().await;
        assert!(matches!(state.phase, DownloadPhase::Paused {}), "Model should remain Paused after relaunch");
        assert!(state.progress() >= paused_progress, "Progress should not decrease after relaunch");
        let model_dir = test_storage_2.config.cache_model_path(&test_storage_2.model(0)).unwrap();
        if model_dir.exists() {
            let mut entries = tokio::fs::read_dir(&model_dir).await?;
            let mut files_after = Vec::new();
            while let Some(entry) = entries.next_entry().await? {
                let file_name = entry.file_name();
                let name_str = file_name.to_string_lossy().to_string();
                if !name_str.starts_with('.') && !name_str.ends_with(".resume_data") {
                    let metadata = entry.metadata().await?;
                    files_after.push((name_str.clone(), metadata.len()));
                }
            }
            for (name, size) in &complete_files {
                assert!(
                    files_after.iter().any(|(n, s)| n == name && s == size),
                    "Complete file {} should remain after relaunch",
                    name
                );
            }
        }
        let model = storage_2.get(&test_storage_2.model(0).identifier()).await.ok_or("Model not found")?;
        model.download().await?;
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let model = storage_2.get(&test_storage_2.model(0).identifier()).await.ok_or("Model not found")?;
        let state = model.state().await;
        assert!(matches!(state.phase, DownloadPhase::Downloading {}), "Should be Downloading after resume");
    }
    Ok(())
}

#[ignore]
#[tokio::test(flavor = "multi_thread")]
async fn test_storage_comprehensive_state_delete_from_various_states() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let base_path = temp_dir.path().to_path_buf();

    let test_storage = TestStorage::new_with_base_path(base_path).await?;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;

    model.download().await?;
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;

    model.pause().await?;
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
    let state = model.state().await;
    assert!(matches!(state.phase, DownloadPhase::Paused {}), "Should be Paused");
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;

    model.cancel().await?;
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let model_dir = test_storage.config.cache_model_path(&test_storage.model(0)).unwrap();
    assert!(
        !model_dir.exists() || {
            let mut entries = tokio::fs::read_dir(&model_dir).await?;
            let mut has_files = false;
            while let Some(entry) = entries.next_entry().await? {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.starts_with('.') {
                    has_files = true;
                    break;
                }
            }
            !has_files
        },
        "Model directory should be empty or removed after delete"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_comprehensive_state_multiple_pause_resume_cycles() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let base_path = temp_dir.path().to_path_buf();

    let test_storage = TestStorage::new_with_base_path(base_path).await?;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;

    model.download().await?;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    for cycle in 1..=3 {
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
        model.pause().await?;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
        let state = model.state().await;
        let paused_progress = state.progress();
        assert!(
            matches!(
                state.phase,
                DownloadPhase::Paused {} | DownloadPhase::Downloaded {} | DownloadPhase::Downloading {}
            ),
            "Should be Paused/Downloaded/Downloading in cycle {} (got {:?})",
            cycle,
            state.phase
        );
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;

        model.download().await?;
        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
        let state = model.state().await;
        assert!(state.progress() >= paused_progress, "Progress should not decrease in cycle {}", cycle);
    }
    Ok(())
}

#[ignore]
#[tokio::test(flavor = "multi_thread")]
async fn test_storage_comprehensive_state_completed_files_preserved_on_relaunch()
-> Result<(), Box<dyn std::error::Error>> {
    {
        let temp_dir = tempfile::tempdir()?;
        let base_path = temp_dir.path().to_path_buf();

        let test_storage = TestStorage::new_with_base_path(base_path.clone()).await?;
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
        model.download().await?;
        for _ in 0..30 {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            let model_dir = test_storage.config.cache_model_path(&test_storage.model(0)).unwrap();
            if model_dir.exists() {
                let mut entries = tokio::fs::read_dir(&model_dir).await?;
                let mut complete_count = 0;
                let mut incomplete_count = 0;
                while let Some(entry) = entries.next_entry().await? {
                    let file_name = entry.file_name();
                    let name_str = file_name.to_string_lossy();
                    if name_str.ends_with(".resume_data") || name_str.starts_with('.') {
                        continue;
                    }
                    let metadata = entry.metadata().await?;
                    let size = metadata.len();
                    if name_str == "config.json" && size == 2907 {
                        complete_count += 1;
                    } else if name_str == "tokenizer.json" && size == 12807982 {
                        complete_count += 1;
                    } else if name_str == "tokenizer_config.json" && size == 16709 {
                        complete_count += 1;
                    } else if size > 0 {
                        incomplete_count += 1;
                    }
                }
                if complete_count > 0 && incomplete_count > 0 {
                    break;
                }
            }
        }
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
        model.pause().await?;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let model_dir = test_storage.config.cache_model_path(&test_storage.model(0)).unwrap();
        let mut files_before = Vec::new();
        if model_dir.exists() {
            let mut entries = tokio::fs::read_dir(&model_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let file_name = entry.file_name();
                let name_str = file_name.to_string_lossy().to_string();
                if !name_str.starts_with('.') && !name_str.ends_with(".resume_data") {
                    let metadata = entry.metadata().await?;
                    files_before.push((name_str.clone(), metadata.len()));
                }
            }
        }
        drop(test_storage.storage);
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let test_storage_2 = TestStorage::new_with_base_path(base_path).await?;
        let model_dir = test_storage_2.config.cache_model_path(&test_storage_2.model(0)).unwrap();
        let mut files_after = Vec::new();
        if model_dir.exists() {
            let mut entries = tokio::fs::read_dir(&model_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let file_name = entry.file_name();
                let name_str = file_name.to_string_lossy().to_string();
                if !name_str.starts_with('.') && !name_str.ends_with(".resume_data") {
                    let metadata = entry.metadata().await?;
                    files_after.push((name_str.clone(), metadata.len()));
                }
            }
        }
        for (name, size) in &files_before {
            assert!(
                files_after.iter().any(|(n, s)| n == name && s == size),
                "Complete file {} should be preserved after relaunch",
                name
            );
        }
        let model = test_storage_2.storage.get(&test_storage_2.model(0).identifier()).await.ok_or("Model not found")?;
        let state = model.state().await;
        assert!(matches!(state.phase, DownloadPhase::Paused {}), "Model should be Paused after relaunch");
    }
    Ok(())
}

#[ignore]
#[tokio::test(flavor = "multi_thread")]
async fn test_storage_comprehensive_state_idempotent_pause() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let base_path = temp_dir.path().to_path_buf();

    let test_storage = TestStorage::new_with_base_path(base_path).await?;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;

    model.download().await?;
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;

    model.pause().await?;
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
    let state = model.state().await;
    let progress_after_first_pause = state.progress();
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;

    model.pause().await?;
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
    let state = model.state().await;
    assert!(matches!(state.phase, DownloadPhase::Paused {}), "Should remain Paused after second pause");
    assert_eq!(state.progress(), progress_after_first_pause, "Progress should not change on idempotent pause");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_comprehensive_state_crc_validation_on_init() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let base_path = temp_dir.path().to_path_buf();

    let test_storage = TestStorage::new_with_base_path(base_path.clone()).await?;
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
    model.download().await?;
    for _ in 0..60 {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        let model_dir = test_storage.config.cache_model_path(&test_storage.model(0)).unwrap();
        if model_dir.exists() {
            let config_path = model_dir.join("config.json");
            let tokenizer_path = model_dir.join("tokenizer.json");
            if config_path.exists() && tokenizer_path.exists() {
                if let Ok(config_meta) = tokio::fs::metadata(&config_path).await {
                    if let Ok(tok_meta) = tokio::fs::metadata(&tokenizer_path).await {
                        if config_meta.len() == 2907 && tok_meta.len() == 12807982 {
                            break;
                        }
                    }
                }
            }
        }
    }
    // Check if download completed before we can pause
    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await;
    if let Some(m) = model {
        let state = m.state().await;
        if !matches!(state.phase, DownloadPhase::Downloaded {}) {
            let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
            model.pause().await?;
        }
    }
    drop(test_storage.storage);

    {
        let test_storage_2 = TestStorage::new_with_base_path(base_path).await?;
        let model_dir = test_storage_2.config.cache_model_path(&test_storage_2.model(0)).unwrap();
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_resume = model_dir.join("config.json.resume_data");
        let tokenizer_resume = model_dir.join("tokenizer.json.resume_data");
        assert!(config_path.exists(), "config.json should exist after relaunch");
        assert!(tokenizer_path.exists(), "tokenizer.json should exist after relaunch");
        assert!(!config_resume.exists(), "config.json.resume_data should be deleted for complete file");
        assert!(!tokenizer_resume.exists(), "tokenizer.json.resume_data should be deleted for complete file");
        let model = test_storage_2.storage.get(&test_storage_2.model(0).identifier()).await.ok_or("Model not found")?;
        let state = model.state().await;
        assert!(
            matches!(
                state.phase,
                DownloadPhase::Paused {} | DownloadPhase::Downloading {} | DownloadPhase::Downloaded {}
            ),
            "Model should be in a valid state after relaunch (Paused, Downloading, or Downloaded)"
        );
    }
    Ok(())
}

#[ignore]
#[tokio::test(flavor = "multi_thread")]
async fn test_storage_comprehensive_state_progress_calculation_with_mixed_files()
-> Result<(), Box<dyn std::error::Error>> {
    {
        let temp_dir = tempfile::tempdir()?;
        let base_path = temp_dir.path().to_path_buf();

        let test_storage = TestStorage::new_with_base_path(base_path).await?;
        let expected_config_bytes = 2907;
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
        model.download().await?;
        let mut config_completed = false;
        for _ in 0..30 {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            let model_dir = test_storage.config.cache_model_path(&test_storage.model(0)).unwrap();
            let config_path = model_dir.join("config.json");
            if let Ok(meta) = tokio::fs::metadata(&config_path).await {
                if meta.len() == expected_config_bytes {
                    config_completed = true;
                    break;
                }
            }
        }
        if config_completed {
            let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
            model.pause().await?;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.ok_or("Model not found")?;
            let state = model.state().await;
            let downloaded_bytes = state.downloaded_bytes;
            assert!(
                downloaded_bytes >= expected_config_bytes as i64,
                "Downloaded bytes should include completed config.json"
            );
        }
    }
    Ok(())
}
