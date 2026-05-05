use std::fs;

use crate::{LockFileInfo, LockFileState, Path, TokioHandle, chrono};

const LOCK_TIMEOUT_MINUTES: i64 = 30;

/// Check if lock file exists and determine its state
pub fn check_lock_file(
    lock_path: &Path,
    our_manager_id: &str,
    our_process_id: u32,
) -> LockFileState {
    if !lock_path.exists() {
        return LockFileState::Missing;
    }

    let lock_info = match read_lock_file(lock_path) {
        Ok(info) => info,
        Err(e) => {
            tracing::warn!("[LOCK] Failed to read lock file {}: {}. Treating as stale.", lock_path.display(), e);
            // Corrupted lock file - treat as stale with dummy info
            return LockFileState::Stale(LockFileInfo {
                manager_id: "unknown".to_string(),
                acquired_at: chrono::Utc::now(),
                process_id: 0,
            });
        },
    };

    // Check if we own this lock (same manager_id + same PID)
    if lock_info.manager_id == our_manager_id {
        if lock_info.process_id == our_process_id {
            return LockFileState::OwnedByUs(lock_info);
        } else {
            // Same app, different process - check if old process is dead
            if is_process_alive(lock_info.process_id) {
                // Old process still alive - this shouldn't happen in normal operation
                tracing::warn!(
                    "[LOCK] Same manager_id but different PID and process is alive: {}",
                    lock_info.process_id
                );
                return LockFileState::OwnedByOtherApp(lock_info);
            } else {
                return LockFileState::OwnedBySameAppOldProcess(lock_info);
            }
        }
    }

    // Different manager_id - check if stale
    if is_lock_stale(&lock_info) {
        return LockFileState::Stale(lock_info);
    }

    LockFileState::OwnedByOtherApp(lock_info)
}

/// Read and parse lock file
fn read_lock_file(lock_path: &Path) -> Result<LockFileInfo, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(lock_path)?;
    let info: LockFileInfo = serde_json::from_str(&contents)?;
    Ok(info)
}

/// Check if a lock is stale based on timeout
fn is_lock_stale(lock_info: &LockFileInfo) -> bool {
    let elapsed = chrono::Utc::now() - lock_info.acquired_at;
    elapsed > chrono::Duration::minutes(LOCK_TIMEOUT_MINUTES)
}

/// Check if a process is still alive (Unix-specific)
#[cfg(unix)]
fn is_process_alive(pid: u32) -> bool {
    use std::process::{Command, Stdio};
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Check if a process is still alive (Windows-specific)
#[cfg(windows)]
fn is_process_alive(pid: u32) -> bool {
    use std::process::{Command, Stdio};
    Command::new("tasklist")
        .args(["/FI", &format!("PID eq {}", pid)])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).contains(&pid.to_string()))
        .unwrap_or(false)
}

/// Acquire lock atomically
pub async fn acquire_lock(
    tokio_handle: &TokioHandle,
    lock_path: &Path,
    manager_id: &str,
) -> Result<(), std::io::Error> {
    let lock_info = LockFileInfo::new(manager_id.to_string(), std::process::id());

    let json =
        serde_json::to_string_pretty(&lock_info).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let lock_path_buf = lock_path.to_path_buf();
    let lock_path_clone = lock_path_buf.clone();
    tokio_handle
        .spawn_blocking(move || {
            // Create parent directory if needed
            if let Some(parent) = lock_path_buf.parent() {
                fs::create_dir_all(parent)?;
            }

            // Atomic write using temp file + rename
            let temp_path = lock_path_buf.with_extension("lock.tmp");
            fs::write(&temp_path, json)?;
            fs::rename(temp_path, &lock_path_buf)?;
            Ok::<(), std::io::Error>(())
        })
        .await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))??;

    tracing::info!("[LOCK] Acquired lock for {} (manager_id: {})", lock_path_clone.display(), manager_id);

    Ok(())
}

/// Release the lock by deleting the lock file
pub async fn release_lock(
    tokio_handle: &TokioHandle,
    lock_path: &Path,
) -> Result<(), std::io::Error> {
    let lock_path = lock_path.to_path_buf();
    tokio_handle
        .spawn_blocking(move || {
            if lock_path.exists() {
                fs::remove_file(&lock_path)?;
                tracing::info!("[LOCK] Released lock: {}", lock_path.display());
            }
            Ok::<(), std::io::Error>(())
        })
        .await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))??;
    Ok(())
}

/// Try to acquire a lock (non-blocking check first)
pub async fn try_acquire_lock(
    tokio_handle: &TokioHandle,
    lock_path: &Path,
    manager_id: &str,
) -> Result<bool, std::io::Error> {
    let lock_path_buf = lock_path.to_path_buf();
    let manager_id_str = manager_id.to_string();
    let process_id = std::process::id();

    let lock_state = tokio_handle
        .spawn_blocking(move || check_lock_file(&lock_path_buf, &manager_id_str, process_id))
        .await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    if lock_state.is_conflict() {
        return Ok(false);
    }
    acquire_lock(tokio_handle, lock_path, manager_id).await?;
    Ok(true)
}
