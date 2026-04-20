mod common;

use std::time::Duration;

use common::TestDownloadManager;
use download_manager::{
    FileCheck, FileDownloadManagerType, LockFileInfo, LockFileState, acquire_lock, check_lock_file, release_lock,
};
use tokio::time::sleep;

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_manager_lock_file_prevents_concurrent_downloads() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("[TEST] ========== LOCK FILE CONCURRENT DOWNLOAD TEST ==========");

    // Create two managers (simulating two app instances)
    let test_manager1 = TestDownloadManager::new(
        "test_apple_manager_lock_file_prevents_concurrent_downloads_1",
        FileDownloadManagerType::Apple,
    )
    .await?;
    let test_manager2 = TestDownloadManager::new(
        "test_apple_manager_lock_file_prevents_concurrent_downloads_2",
        FileDownloadManagerType::Apple,
    )
    .await?;

    let dest_path = test_manager1.dest_path("locked_file.safetensors");
    tracing::info!("[TEST] Destination: {}", dest_path.display());

    // Clean up any existing artifacts
    let _ = std::fs::remove_file(&dest_path);
    let _ = std::fs::remove_file(format!("{}.crc", dest_path.display()));
    let _ = std::fs::remove_file(format!("{}.resume_data", dest_path.display()));
    let _ = std::fs::remove_file(format!("{}.lock", dest_path.display()));

    // Manager 1 starts download (should acquire lock)
    tracing::info!("[TEST] Manager 1: Creating task and starting download...");
    let task1 = test_manager1
        .manager
        .file_download_task(&test_manager1.test_file.url, &dest_path, FileCheck::None, None)
        .await?;

    task1.download().await?;
    tracing::info!("[TEST] Manager 1: Download started");

    // Give it a moment to acquire the lock
    sleep(Duration::from_millis(500)).await;

    // Verify lock file exists
    let lock_path = format!("{}.lock", dest_path.display());
    assert!(std::path::Path::new(&lock_path).exists(), "Lock file should exist after download started");

    // Manager 2 creates the same task; reduction should mark it as LockedByOther
    tracing::info!("[TEST] Manager 2: Creating task for same file...");
    let task2 = test_manager2
        .manager
        .file_download_task(&test_manager2.test_file.url, &dest_path, FileCheck::None, None)
        .await?;

    // Wait a moment for state to be computed
    sleep(Duration::from_millis(200)).await;

    let state2 = task2.state().await;
    match state2.phase {
        download_manager::FileDownloadPhase::LockedByOther(owner) => {
            tracing::info!("[TEST] ✓ Task 2 marked as locked by {}", owner);
            assert!(owner.contains("downloads_1"));
        },
        other => panic!("Task 2 should be LockedByOther, got: {:?}", other),
    }

    // Cancel manager 1's download
    tracing::info!("[TEST] Manager 1: Cancelling download...");
    task1.cancel().await?;

    tracing::info!("[TEST] ========== LOCK FILE TEST PASSED ==========");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lock_file_stale_detection() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("[TEST] ========== LOCK FILE STALE DETECTION TEST ==========");

    let test_manager =
        TestDownloadManager::new("test_apple_manager_lock_file_stale_detection", FileDownloadManagerType::Apple)
            .await?;
    let dest_path = test_manager.dest_path("stale_lock_file.txt");
    let lock_path = std::path::PathBuf::from(format!("{}.lock", dest_path.display()));

    // Clean up
    let _ = std::fs::remove_file(&dest_path);
    let _ = std::fs::remove_file(&lock_path);

    // Create a stale lock file (with a fake PID that doesn't exist)
    let fake_lock = LockFileInfo {
        manager_id: "com.otherapp.trymirai.url-session-download-manager".to_string(),
        acquired_at: chrono::Utc::now() - chrono::Duration::hours(2), // 2 hours ago (stale)
        process_id: 99999,                                            // Non-existent PID
    };

    let lock_json = serde_json::to_string_pretty(&fake_lock)?;
    std::fs::write(&lock_path, lock_json)?;
    tracing::info!("[TEST] Created fake stale lock file");

    // Check lock state - should be detected as stale
    let our_manager_id = test_manager.manager.manager_id();
    let lock_state = check_lock_file(&lock_path, our_manager_id, std::process::id());

    tracing::info!("[TEST] Lock state: {:?}", lock_state);

    match lock_state {
        LockFileState::Stale(info) => {
            tracing::info!("[TEST] ✓ Lock correctly detected as stale");
            assert_eq!(info.process_id, 99999);
        },
        _ => {
            panic!("Lock should have been detected as stale, got: {:?}", lock_state);
        },
    }

    // Now acquire the lock (should succeed)
    tracing::info!("[TEST] Attempting to acquire stale lock...");
    let handle = tokio::runtime::Handle::current();
    acquire_lock(&handle, &lock_path, our_manager_id).await?;
    tracing::info!("[TEST] ✓ Successfully acquired stale lock");

    // Verify we own it now
    let lock_state = check_lock_file(&lock_path, our_manager_id, std::process::id());
    match lock_state {
        LockFileState::OwnedByUs(_) => {
            tracing::info!("[TEST] ✓ Lock now owned by us");
        },
        _ => {
            panic!("Lock should be owned by us, got: {:?}", lock_state);
        },
    }

    // Release the lock
    let handle = tokio::runtime::Handle::current();
    release_lock(&handle, &lock_path).await?;
    assert!(!lock_path.exists(), "Lock file should be deleted after release");

    tracing::info!("[TEST] ========== STALE DETECTION TEST PASSED ==========");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_manager_lock_file_same_app_crash_recovery() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("[TEST] ========== LOCK FILE CRASH RECOVERY TEST ==========");

    let test_manager = TestDownloadManager::new(
        "test_apple_manager_lock_file_same_app_crash_recovery",
        FileDownloadManagerType::Apple,
    )
    .await?;
    let dest_path = test_manager.dest_path("crash_recovery_file.txt");
    let lock_path = std::path::PathBuf::from(format!("{}.lock", dest_path.display()));

    // Clean up
    let _ = std::fs::remove_file(&dest_path);
    let _ = std::fs::remove_file(&lock_path);

    let our_manager_id = test_manager.manager.manager_id();

    // Simulate a crash: create a lock with our manager_id but different PID
    let old_lock = LockFileInfo {
        manager_id: our_manager_id.to_string(),
        acquired_at: chrono::Utc::now() - chrono::Duration::minutes(5),
        process_id: std::process::id() + 10000, // Different PID from same app
    };

    let lock_json = serde_json::to_string_pretty(&old_lock)?;
    std::fs::write(&lock_path, lock_json)?;
    tracing::info!("[TEST] Created fake lock from crashed process (same app)");

    // Check lock state - should be detected as OwnedBySameAppOldProcess (if PID dead)
    // or OwnedByUs if we can detect it's safe to reacquire
    let lock_state = check_lock_file(&lock_path, our_manager_id, std::process::id());

    tracing::info!("[TEST] Lock state after crash: {:?}", lock_state);

    // The lock should either be Stale or OwnedBySameAppOldProcess
    // Both mean we can reacquire it
    assert!(lock_state.can_proceed(), "Lock from crashed same-app process should be reacquirable");

    // Acquire the lock (simulating app restart after crash)
    let handle = tokio::runtime::Handle::current();
    acquire_lock(&handle, &lock_path, our_manager_id).await?;
    tracing::info!("[TEST] ✓ Successfully reacquired lock after simulated crash");

    // Verify we own it now with current PID
    let lock_state = check_lock_file(&lock_path, our_manager_id, std::process::id());
    match lock_state {
        LockFileState::OwnedByUs(info) => {
            tracing::info!("[TEST] ✓ Lock owned by current process");
            assert_eq!(info.process_id, std::process::id());
            assert_eq!(info.manager_id, our_manager_id);
        },
        _ => {
            panic!("Lock should be owned by us, got: {:?}", lock_state);
        },
    }

    let handle = tokio::runtime::Handle::current();
    release_lock(&handle, &lock_path).await?;

    tracing::info!("[TEST] ========== CRASH RECOVERY TEST PASSED ==========");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_manager_lock_file_atomic_operations() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("[TEST] ========== LOCK FILE ATOMIC OPERATIONS TEST ==========");

    let test_manager =
        TestDownloadManager::new("test_apple_manager_lock_file_atomic_operations", FileDownloadManagerType::Apple)
            .await?;
    let dest_path = test_manager.dest_path("atomic_lock_file.txt");
    let lock_path = std::path::PathBuf::from(format!("{}.lock", dest_path.display()));

    // Clean up
    let _ = std::fs::remove_file(&dest_path);
    let _ = std::fs::remove_file(&lock_path);

    let manager_id = test_manager.manager.manager_id();

    // Test acquire
    tracing::info!("[TEST] Acquiring lock...");
    let handle = tokio::runtime::Handle::current();
    acquire_lock(&handle, &lock_path, manager_id).await?;
    assert!(lock_path.exists(), "Lock file should exist after acquire");

    // Read and verify lock contents
    let lock_contents = std::fs::read_to_string(&lock_path)?;
    let lock_info: LockFileInfo = serde_json::from_str(&lock_contents)?;

    assert_eq!(lock_info.manager_id, manager_id);
    assert_eq!(lock_info.process_id, std::process::id());
    tracing::info!("[TEST] ✓ Lock file contains correct information");

    // Test release
    tracing::info!("[TEST] Releasing lock...");
    let handle = tokio::runtime::Handle::current();
    release_lock(&handle, &lock_path).await?;
    assert!(!lock_path.exists(), "Lock file should be deleted after release");

    tracing::info!("[TEST] ========== ATOMIC OPERATIONS TEST PASSED ==========");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_manager_lock_file_removed_after_download_completion() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("[TEST] ========== LOCK FILE CLEANUP TEST ==========");

    // Create a test manager
    let test_manager = TestDownloadManager::new(
        "test_apple_manager_lock_file_removed_after_download_completion",
        FileDownloadManagerType::Apple,
    )
    .await?;
    let dest_path = test_manager.dest_path("test_file");
    let lock_path = std::path::PathBuf::from(format!("{}.lock", dest_path.display()));

    tracing::info!("[TEST] Destination: {}", dest_path.display());
    tracing::info!("[TEST] Lock path: {}", lock_path.display());

    // Clean up any existing artifacts
    let _ = std::fs::remove_file(&dest_path);
    let _ = std::fs::remove_file(format!("{}.crc", dest_path.display()));
    let _ = std::fs::remove_file(format!("{}.resume_data", dest_path.display()));
    let _ = std::fs::remove_file(&lock_path);

    // Start download (should acquire lock)
    tracing::info!("[TEST] Starting download...");
    let task =
        test_manager.manager.file_download_task(&test_manager.test_file.url, &dest_path, FileCheck::None, None).await?;

    task.download().await?;
    tracing::info!("[TEST] Download started");

    // Wait a moment for lock acquisition
    sleep(Duration::from_millis(200)).await;

    // Verify lock file exists during download
    assert!(lock_path.exists(), "Lock file should exist during download");
    tracing::info!("[TEST] ✓ Lock file exists during download");

    // Wait for download to complete (with timeout)
    let mut attempts = 0;
    let max_attempts = 600; // 60 seconds max
    let mut download_completed = false;

    while attempts < max_attempts {
        let state = task.state().await;
        tracing::debug!("[TEST] Download state: {:?}", state.phase);

        if matches!(state.phase, download_manager::FileDownloadPhase::Downloaded) {
            download_completed = true;
            break;
        }

        if matches!(state.phase, download_manager::FileDownloadPhase::Error(_)) {
            panic!("Download failed: {:?}", state.phase);
        }

        sleep(Duration::from_millis(100)).await;
        attempts += 1;
    }

    assert!(download_completed, "Download should complete within timeout");
    tracing::info!("[TEST] ✓ Download completed successfully");

    // Give a small delay for lock cleanup to complete
    sleep(Duration::from_millis(200)).await;

    // Verify lock file is removed after completion
    assert!(!lock_path.exists(), "Lock file should be removed after successful download completion");
    tracing::info!("[TEST] ✓ Lock file removed after download completion");

    tracing::info!("[TEST] ========== LOCK FILE CLEANUP TEST PASSED ==========");
    Ok(())
}
