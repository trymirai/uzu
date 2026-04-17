#[cfg(target_vendor = "apple")]
mod common;

use common::test_helpers::TestStorage;
use nagare::storage::types::DownloadPhase;

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_debug_file_preservation_complete_app_relaunch_resume() -> Result<(), Box<dyn std::error::Error>> {
    // Enable test tracing to file for deeper diagnostics
    common::tracing_setup::init_test_tracing();
    let temp_dir = tempfile::tempdir()?;
    let base_path = temp_dir.path().to_path_buf();
    let base_path_clone = base_path.clone();

    let (progress_session1, _bytes_session1, _total_bytes);
    {
        let (p1, b1, tb) = std::thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(async {
                let test_storage = TestStorage::new_with_base_path(base_path_clone).await.unwrap();
                let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
                tracing::info!("[S1] Starting download for {}", test_storage.model(0).identifier());
                model.download().await.unwrap();
                let mut reached_target = false;
                for i in 0..180 {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
                    tracing::info!("[S1] Iteration {}: pausing model", i);
                    model.pause().await.unwrap();
                    let state = model.state().await;
                    let progress = state.progress();
                    tracing::info!(
                        "[S1] Iteration {}: phase={:?}, progress={:.2}%, bytes={}/{}",
                        i,
                        state.phase,
                        progress * 100.0,
                        state.downloaded_bytes,
                        state.total_bytes
                    );

                    if progress >= 0.08 && progress <= 0.25 {
                        reached_target = true;
                        break;
                    }
                    model.download().await.unwrap();
                }
                assert!(reached_target, "Failed to reach target progress");
                let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
                tracing::info!("[S1] Pausing before shutdown");
                model.pause().await.unwrap();
                tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
                let state = model.state().await;
                let progress = state.progress();
                let bytes = state.downloaded_bytes;
                let total = state.total_bytes;
                tracing::info!(
                    "[S1] Pre-shutdown state: phase={:?}, progress={:.2}%, bytes={}/{}",
                    state.phase,
                    progress * 100.0,
                    bytes,
                    total
                );
                let model_dir = test_storage.config.cache_model_path(&test_storage.model(0)).unwrap();
                let mut entries = tokio::fs::read_dir(&model_dir).await.unwrap();
                while let Some(entry) = entries.next_entry().await.unwrap() {
                    let name = entry.file_name();
                    let meta = entry.metadata().await.unwrap();
                    let marker = if name.to_string_lossy().ends_with(".resume_data") {
                        "📄"
                    } else {
                        "📦"
                    };
                    tracing::info!("[S1] Model dir entry {} {} ({} bytes)", marker, name.to_string_lossy(), meta.len());
                }
                drop(test_storage.storage);
                (progress, bytes, total)
            })
        })
        .join()
        .expect("Session 1 thread panicked");
        progress_session1 = p1;
        _bytes_session1 = b1;
        _total_bytes = tb;
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    {
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(async {
                common::tracing_setup::init_test_tracing();
                let test_storage = TestStorage::new_with_base_path(base_path).await.unwrap();
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;

                let mut entries =
                    tokio::fs::read_dir(&test_storage.config.cache_model_path(&test_storage.model(0)).unwrap())
                        .await
                        .unwrap();
                while let Some(entry) = entries.next_entry().await.unwrap() {
                    let name = entry.file_name();
                    let meta = entry.metadata().await.unwrap();
                    tracing::info!(
                        "[S2] Existing entry before init: {} ({} bytes)",
                        name.to_string_lossy(),
                        meta.len()
                    );
                }

                let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
                let state = model.state().await;
                let progress_session2 = state.progress();
                let bytes_session2 = state.downloaded_bytes;
                tracing::info!(
                    "[S2] Post-init state: phase={:?}, progress={:.2}%, bytes={}/{}",
                    state.phase,
                    progress_session2 * 100.0,
                    state.downloaded_bytes,
                    state.total_bytes
                );
                let progress_diff = (progress_session1 - progress_session2).abs();
                assert!(
                    progress_diff < 0.02,
                    "Progress NOT preserved across sessions! Session1: {:.1}%, Session2: {:.1}%",
                    progress_session1 * 100.0,
                    progress_session2 * 100.0
                );
                tracing::info!("[S2] Resuming download after relaunch");
                model.download().await.unwrap();
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
                let state = model.state().await;
                let progress_after_resume = state.progress();
                let bytes_after_resume = state.downloaded_bytes;
                tracing::info!(
                    "[S2] After resume: phase={:?}, progress={:.2}%, bytes={}/{} (delta bytes: {})",
                    state.phase,
                    progress_after_resume * 100.0,
                    bytes_after_resume,
                    state.total_bytes,
                    bytes_after_resume.saturating_sub(bytes_session2)
                );
                if bytes_after_resume >= bytes_session2 * 90 / 100 {
                } else {
                }
                drop(test_storage);
            })
        })
        .join()
        .expect("Session 2 thread panicked");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_debug_file_preservation_resume_from_20_percent() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let base_path = temp_dir.path().to_path_buf();
    let base_path_clone = base_path.clone();

    let (progress_at_pause, bytes_at_pause, _total_bytes);
    {
        let test_storage = TestStorage::new_with_base_path(base_path_clone).await.unwrap();
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
        model.download().await.unwrap();
        let mut reached_target = false;
        for i in 0..180 {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
            let state = model.state().await;
            let progress = state.progress();
            let _downloaded = state.downloaded_bytes;
            let _total = state.total_bytes;
            if i % 4 == 0 {}
            if progress >= 0.08 && progress <= 0.25 {
                reached_target = true;
                break;
            }
        }
        if !reached_target {
            return Err("Failed to reach target progress range in time".into());
        }
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
        model.pause().await?;
        for i in 0..20 {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
            let state = model.state().await;
            if matches!(state.phase, DownloadPhase::Paused) {
                break;
            }
            if i == 19 {}
        }
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
        let state = model.state().await;
        progress_at_pause = state.progress();
        bytes_at_pause = state.downloaded_bytes;
        _total_bytes = state.total_bytes;
        if !matches!(state.phase, DownloadPhase::Paused) {}
        let model_dir = test_storage.config.cache_model_path(&test_storage.model(0)).unwrap();
        let mut resume_files = Vec::new();
        if model_dir.exists() {
            let mut entries = tokio::fs::read_dir(&model_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let name = entry.file_name();
                let name_str = name.to_string_lossy().to_string();
                if name_str.ends_with(".resume_data") {
                    let meta = entry.metadata().await?;
                    resume_files.push((name_str.clone(), meta.len()));
                    let resume_path = model_dir.join(&name_str);
                    if let Ok(data) = tokio::fs::read(&resume_path).await {
                        if let Ok(_resume) =
                            download_manager::managers::apple::URLSessionDownloadTaskResumeData::from_bytes(&data)
                        {
                        }
                    }
                }
            }
        }
        assert!(!resume_files.is_empty(), "Should have at least one .resume_data file");
        if model_dir.exists() {
            let mut entries = tokio::fs::read_dir(&model_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let _name = entry.file_name();
                let _meta = entry.metadata().await?;
            }
        }
        drop(test_storage);
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    {
        let test_storage = TestStorage::new_with_base_path(base_path).await.unwrap();
        let model_dir = test_storage.config.cache_model_path(&test_storage.model(0)).unwrap();
        if model_dir.exists() {
            let mut entries = tokio::fs::read_dir(&model_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let _name = entry.file_name();
                let _meta = entry.metadata().await?;
            }
        }
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
        let state = model.state().await;
        let progress_after_relaunch = state.progress();
        let bytes_after_relaunch = state.downloaded_bytes;
        assert!(matches!(state.phase, DownloadPhase::Paused), "Model should be Paused after relaunch");
        let progress_diff = (progress_at_pause - progress_after_relaunch).abs();
        let _bytes_diff = (bytes_at_pause as i64 - bytes_after_relaunch as i64).abs();
        assert!(
            progress_diff < 0.10,
            "Progress should be preserved after relaunch (pause: {:.1}%, relaunch: {:.1}%, diff: {:.1}%)",
            progress_at_pause * 100.0,
            progress_after_relaunch * 100.0,
            progress_diff * 100.0
        );
        let mut resume_files_after = Vec::new();
        if model_dir.exists() {
            let mut entries = tokio::fs::read_dir(&model_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let name = entry.file_name();
                let name_str = name.to_string_lossy().to_string();
                if name_str.ends_with(".resume_data") {
                    let meta = entry.metadata().await?;
                    resume_files_after.push((name_str.clone(), meta.len()));
                    let resume_path = model_dir.join(&name_str);
                    if let Ok(data) = tokio::fs::read(&resume_path).await {
                        if let Ok(_resume) =
                            download_manager::managers::apple::URLSessionDownloadTaskResumeData::from_bytes(&data)
                        {
                        }
                    }
                }
            }
        }
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
        model.download().await?;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let model_immediate = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
        let state = model_immediate.state().await;
        let progress_immediate = state.progress();
        let bytes_immediate = state.downloaded_bytes;
        if bytes_immediate < bytes_after_relaunch * 90 / 100 {
            panic!(
                "\n   ❌ CRITICAL: Download restarted from scratch on resume!\n\
                   Before resume: {} bytes ({:.1}%)\n\
                   After resume: {} bytes ({:.1}%)\n\
                   Loss: {} bytes",
                bytes_after_relaunch,
                progress_after_relaunch * 100.0,
                bytes_immediate,
                progress_immediate * 100.0,
                bytes_after_relaunch - bytes_immediate
            );
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
        model.pause().await?;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
        let state = model.state().await;
        let progress_after_resume = state.progress();
        let bytes_after_resume = state.downloaded_bytes;
        if model_dir.exists() {
            let mut entries = tokio::fs::read_dir(&model_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let _name = entry.file_name();
                let _meta = entry.metadata().await?;
            }
        }
        assert!(
            bytes_after_resume >= bytes_after_relaunch,
            "Progress should not decrease after resuming and downloading (relaunch: {:.1}%, after 1s: {:.1}%)",
            progress_after_relaunch * 100.0,
            progress_after_resume * 100.0
        );
        drop(test_storage);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_debug_file_preservation_completed_file_detection() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let base_dir = temp_dir.path().to_path_buf();
    let base_dir_clone = base_dir.clone();
    let (_progress_after_pause, downloaded_bytes_after_pause);
    {
        let test_storage = TestStorage::new_with_base_path(base_dir_clone).await.unwrap();
        let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
        model.download().await?;
        let mut config_done = false;
        let mut tokenizer_done = false;
        for _ in 0..60 {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            let model_dir = test_storage.config.cache_model_path(&test_storage.model(0)).unwrap();
            let config_path = model_dir.join("config.json");
            let tokenizer_path = model_dir.join("tokenizer.json");
            if !config_done {
                if let Ok(meta) = tokio::fs::metadata(&config_path).await {
                    if meta.len() == 2907 {
                        config_done = true;
                    }
                }
            }
            if !tokenizer_done {
                if let Ok(meta) = tokio::fs::metadata(&tokenizer_path).await {
                    if meta.len() == 12807982 {
                        tokenizer_done = true;
                    }
                }
            }
            if config_done && tokenizer_done {
                break;
            }
        }

        // Check if download is already complete before trying to pause
        let model_before_pause = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();

        let state = model_before_pause.state().await;
        if !matches!(state.phase, DownloadPhase::Downloaded) {
            let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
            model.pause().await?;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }

        let model_paused = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
        let state = model_paused.state().await;
        _progress_after_pause = state.progress();
        downloaded_bytes_after_pause = state.downloaded_bytes;
        drop(test_storage);
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    {
        let test_storage = TestStorage::new_with_base_path(base_dir).await.unwrap();
        let model_dir = test_storage.config.cache_model_path(&test_storage.model(0)).unwrap();
        if model_dir.exists() {
            let mut entries = tokio::fs::read_dir(&model_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let _name = entry.file_name();
                let _meta = entry.metadata().await?;
            }
        }
        let model_relaunched = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
        let state = model_relaunched.state().await;
        let _progress_after_relaunch = state.progress();
        let downloaded_bytes_after_relaunch = state.downloaded_bytes;
        assert!(
            matches!(state.phase, DownloadPhase::Paused | DownloadPhase::Downloaded),
            "Model should be Paused or Downloaded after relaunch, got: {:?}",
            state.phase
        );
        let bytes_diff = (downloaded_bytes_after_pause as i64 - downloaded_bytes_after_relaunch as i64).abs();
        assert!(
            bytes_diff < 1000,
            "Downloaded bytes should match after relaunch (pause: {}, relaunch: {}, diff: {})",
            downloaded_bytes_after_pause,
            downloaded_bytes_after_relaunch,
            bytes_diff
        );
        if model_dir.exists() {
            let mut entries = tokio::fs::read_dir(&model_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let name = entry.file_name();
                let _name_str = name.to_string_lossy();
                let _meta = entry.metadata().await?;
            }
        }

        // Only try to resume if not already downloaded
        let state = model_relaunched.state().await;
        if !matches!(state.phase, DownloadPhase::Downloaded) {
            let model = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
            model.download().await?;
            tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
        }

        let model_resumed = test_storage.storage.get(&test_storage.model(0).identifier()).await.unwrap();
        let state = model_resumed.state().await;
        let _progress_after_resume = state.progress();
        let downloaded_bytes_after_resume = state.downloaded_bytes;
        assert!(
            matches!(state.phase, DownloadPhase::Downloading | DownloadPhase::Paused | DownloadPhase::Downloaded),
            "Model should be Downloading, Paused, or Downloaded after resume, got: {:?}",
            state.phase
        );
        assert!(
            downloaded_bytes_after_resume >= downloaded_bytes_after_relaunch,
            "Downloaded bytes should not decrease after resume (relaunch: {}, resume: {})",
            downloaded_bytes_after_relaunch,
            downloaded_bytes_after_resume
        );
        if model_dir.exists() {
            let mut entries = tokio::fs::read_dir(&model_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let name = entry.file_name();
                let _name_str = name.to_string_lossy();
                let _meta = entry.metadata().await?;
            }
        }
        if downloaded_bytes_after_resume < downloaded_bytes_after_relaunch * 95 / 100 {
            panic!(
                "\n   ❌ FAIL: Download restarted from scratch!\n   Expected: >= {} bytes\n   Got: {} bytes\n   Loss: {} bytes",
                downloaded_bytes_after_relaunch * 95 / 100,
                downloaded_bytes_after_resume,
                downloaded_bytes_after_relaunch - downloaded_bytes_after_resume
            );
        }
        drop(test_storage);
    }
    Ok(())
}
