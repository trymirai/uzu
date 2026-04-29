use std::path::{Path, PathBuf};

use crate::{LockFileInfo, LockFileState};

const LOCK_TIMEOUT_MINUTES: i64 = 30;

pub fn check_lock_file(
    lock_path: &Path,
    our_manager_id: &str,
    our_process_id: u32,
) -> LockFileState {
    if !lock_path.exists() {
        return LockFileState::Missing;
    }

    let lock_info = match read_lock_file(lock_path) {
        Ok(lock_info) => lock_info,
        Err(_) => {
            return LockFileState::Stale(LockFileInfo {
                manager_id: "unknown".to_string(),
                acquired_at: chrono::Utc::now(),
                process_id: 0,
            });
        },
    };

    if lock_info.manager_id == our_manager_id {
        if lock_info.process_id == our_process_id {
            return LockFileState::OwnedByUs(lock_info);
        }

        if is_process_alive(lock_info.process_id) {
            return LockFileState::OwnedByOtherApp(lock_info);
        }

        return LockFileState::OwnedBySameAppOldProcess(lock_info);
    }

    if is_lock_stale(&lock_info) {
        LockFileState::Stale(lock_info)
    } else {
        LockFileState::OwnedByOtherApp(lock_info)
    }
}

pub async fn acquire_lock(
    lock_path: &Path,
    manager_id: &str,
) -> Result<(), std::io::Error> {
    let lock_info = LockFileInfo::new(manager_id.to_string(), std::process::id());
    let lock_contents = serde_json::to_string_pretty(&lock_info).map_err(std::io::Error::other)?;

    if let Some(parent) = lock_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let temporary_lock_path = temporary_lock_path(lock_path);
    tokio::fs::write(&temporary_lock_path, lock_contents).await?;
    tokio::fs::rename(temporary_lock_path, lock_path).await
}

pub async fn release_lock(lock_path: &Path) -> Result<(), std::io::Error> {
    if lock_path.exists() {
        tokio::fs::remove_file(lock_path).await?;
    }
    Ok(())
}

pub async fn try_acquire_lock(
    lock_path: &Path,
    manager_id: &str,
) -> Result<bool, std::io::Error> {
    if check_lock_file(lock_path, manager_id, std::process::id()).is_conflict() {
        return Ok(false);
    }

    acquire_lock(lock_path, manager_id).await?;
    Ok(true)
}

fn read_lock_file(lock_path: &Path) -> Result<LockFileInfo, Box<dyn std::error::Error>> {
    Ok(serde_json::from_str(&std::fs::read_to_string(lock_path)?)?)
}

fn is_lock_stale(lock_info: &LockFileInfo) -> bool {
    chrono::Utc::now() - lock_info.acquired_at > chrono::Duration::minutes(LOCK_TIMEOUT_MINUTES)
}

#[cfg(unix)]
fn is_process_alive(process_id: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &process_id.to_string()])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(windows)]
fn is_process_alive(process_id: u32) -> bool {
    std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {process_id}")])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).contains(&process_id.to_string()))
        .unwrap_or(false)
}

fn temporary_lock_path(lock_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.tmp", lock_path.display()))
}
