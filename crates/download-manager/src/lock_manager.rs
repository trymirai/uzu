use std::path::{Path, PathBuf};

use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::{LockFileInfo, LockFileState};

const LOCK_TIMEOUT_MINUTES: i64 = 30;

#[derive(Debug)]
pub(crate) struct DestinationLockLease {
    lock_path: PathBuf,
    manager_id: String,
    instance_id: Uuid,
}

impl DestinationLockLease {
    pub(crate) async fn acquire_for_destination(
        destination_path: &Path,
        manager_id: &str,
        instance_id: Uuid,
    ) -> Result<Self, std::io::Error> {
        Self::acquire(&lock_path_for_destination(destination_path), manager_id, instance_id).await
    }

    pub(crate) async fn acquire(
        lock_path: &Path,
        manager_id: &str,
        instance_id: Uuid,
    ) -> Result<Self, std::io::Error> {
        acquire_lock(lock_path, manager_id, instance_id).await?;
        Ok(Self {
            lock_path: lock_path.to_path_buf(),
            manager_id: manager_id.to_string(),
            instance_id,
        })
    }

    pub(crate) async fn release(self) -> Result<bool, std::io::Error> {
        release_lock_if_owned(&self.lock_path, &self.manager_id, self.instance_id).await
    }
}

pub(crate) fn lock_path_for_destination(destination: &Path) -> PathBuf {
    PathBuf::from(format!("{}.lock", destination.display()))
}

pub async fn check_lock_file(
    lock_path: &Path,
    our_manager_id: &str,
    our_instance_id: Uuid,
    our_process_id: u32,
) -> LockFileState {
    let bytes = match std::fs::read(lock_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return LockFileState::Missing,
        Err(_) => return classify_unparseable_lock(lock_path, Vec::new()),
    };

    let lock_info = match serde_json::from_slice::<LockFileInfo>(&bytes) {
        Ok(lock_info) => lock_info,
        Err(_) => return classify_unparseable_lock(lock_path, bytes),
    };

    if lock_info.manager_id == our_manager_id {
        if lock_info.process_id == our_process_id {
            if lock_info.instance_id == our_instance_id {
                return LockFileState::OwnedByUs(lock_info);
            }
            return LockFileState::OwnedByOtherApp(lock_info);
        }

        if is_process_alive(lock_info.process_id).await {
            return LockFileState::OwnedByOtherApp(lock_info);
        }

        return LockFileState::OwnedBySameAppOldProcess(lock_info);
    }

    if is_process_alive(lock_info.process_id).await {
        return LockFileState::OwnedByOtherApp(lock_info);
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
    instance_id: Uuid,
) -> Result<(), std::io::Error> {
    let lock_info = LockFileInfo::new(manager_id.to_string(), instance_id, std::process::id());
    let lock_contents = serde_json::to_string_pretty(&lock_info).map_err(std::io::Error::other)?;

    if let Some(parent) = lock_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    match write_new_lock_file(lock_path, lock_contents.as_bytes()).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            match check_lock_file(lock_path, manager_id, instance_id, std::process::id()).await {
                LockFileState::OwnedByUs(_) => Ok(()),
                LockFileState::OwnedBySameAppOldProcess(observed) | LockFileState::Stale(observed) => {
                    match reclaim_stale_lock(lock_path, ReclaimExpectation::Matching(observed)).await? {
                        ReclaimOutcome::Reclaimed | ReclaimOutcome::Missing => {
                            Box::pin(acquire_lock(lock_path, manager_id, instance_id)).await
                        },
                        ReclaimOutcome::Changed => Err(std::io::Error::new(
                            std::io::ErrorKind::AlreadyExists,
                            "destination lock changed while reclaiming stale lock",
                        )),
                    }
                },
                LockFileState::StaleUnparseable(snapshot) => {
                    match reclaim_stale_lock(lock_path, ReclaimExpectation::UnparseableSnapshot(snapshot)).await? {
                        ReclaimOutcome::Reclaimed | ReclaimOutcome::Missing => {
                            Box::pin(acquire_lock(lock_path, manager_id, instance_id)).await
                        },
                        ReclaimOutcome::Changed => Err(std::io::Error::new(
                            std::io::ErrorKind::AlreadyExists,
                            "destination lock changed while reclaiming stale lock",
                        )),
                    }
                },
                LockFileState::OwnedByOtherApp(info) => Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!("destination locked by {}", info.manager_id),
                )),
                LockFileState::Missing => Box::pin(acquire_lock(lock_path, manager_id, instance_id)).await,
            }
        },
        Err(error) => Err(error),
    }
}

async fn write_new_lock_file(
    lock_path: &Path,
    lock_contents: &[u8],
) -> Result<(), std::io::Error> {
    let temporary_lock_path = lock_path.with_extension(format!("lock-tmp-{}", Uuid::new_v4()));
    let temporary_result = async {
        let mut file = tokio::fs::OpenOptions::new().write(true).create_new(true).open(&temporary_lock_path).await?;
        file.write_all(lock_contents).await?;
        file.sync_all().await
    }
    .await;
    if let Err(error) = temporary_result {
        let _ = tokio::fs::remove_file(&temporary_lock_path).await;
        return Err(error);
    }

    match tokio::fs::hard_link(&temporary_lock_path, lock_path).await {
        Ok(()) => {
            let _ = tokio::fs::remove_file(&temporary_lock_path).await;
            Ok(())
        },
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let _ = tokio::fs::remove_file(&temporary_lock_path).await;
            Err(error)
        },
        Err(_) => {
            let result = write_new_lock_file_direct(lock_path, lock_contents).await;
            let _ = tokio::fs::remove_file(&temporary_lock_path).await;
            result
        },
    }
}

async fn write_new_lock_file_direct(
    lock_path: &Path,
    lock_contents: &[u8],
) -> Result<(), std::io::Error> {
    let mut file = tokio::fs::OpenOptions::new().write(true).create_new(true).open(lock_path).await?;
    if let Err(error) = file.write_all(lock_contents).await {
        let _ = tokio::fs::remove_file(lock_path).await;
        return Err(error);
    }
    if let Err(error) = file.sync_all().await {
        let _ = tokio::fs::remove_file(lock_path).await;
        return Err(error);
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) enum ReclaimExpectation {
    Matching(LockFileInfo),
    UnparseableSnapshot(Vec<u8>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReclaimOutcome {
    Reclaimed,
    Missing,
    Changed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RestoreOutcome {
    Restored,
    DestinationAlreadyExists,
}

pub(crate) async fn reclaim_stale_lock(
    lock_path: &Path,
    expectation: ReclaimExpectation,
) -> Result<ReclaimOutcome, std::io::Error> {
    let quarantine_path = lock_path.with_extension(format!("reclaim-{}", Uuid::new_v4()));
    match tokio::fs::rename(lock_path, &quarantine_path).await {
        Ok(()) => {},
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(ReclaimOutcome::Missing),
        Err(error) => return Err(error),
    }
    let confirmed = match &expectation {
        ReclaimExpectation::Matching(observed) => {
            matches!(read_lock_file(&quarantine_path).ok(), Some(actual) if &actual == observed)
        },
        ReclaimExpectation::UnparseableSnapshot(snapshot) => {
            matches!(tokio::fs::read(&quarantine_path).await.ok(), Some(bytes) if &bytes == snapshot)
        },
    };
    if confirmed {
        let _ = tokio::fs::remove_file(&quarantine_path).await;
        return Ok(ReclaimOutcome::Reclaimed);
    }
    let _ = try_restore_quarantine(&quarantine_path, lock_path).await?;
    Ok(ReclaimOutcome::Changed)
}

pub async fn release_lock_if_owned(
    lock_path: &Path,
    manager_id: &str,
    instance_id: Uuid,
) -> Result<bool, std::io::Error> {
    let observed = match check_lock_file(lock_path, manager_id, instance_id, std::process::id()).await {
        LockFileState::OwnedByUs(info) => info,
        _ => return Ok(false),
    };
    let quarantine_path = lock_path.with_extension(format!("release-{}", Uuid::new_v4()));
    match tokio::fs::rename(lock_path, &quarantine_path).await {
        Ok(()) => {},
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    }
    let parsed = read_lock_file(&quarantine_path).ok();
    let confirmed = matches!(parsed.as_ref(), Some(actual) if actual == &observed);
    if confirmed {
        let _ = tokio::fs::remove_file(&quarantine_path).await;
        return Ok(true);
    }
    let _ = try_restore_quarantine(&quarantine_path, lock_path).await?;
    Ok(false)
}

pub(crate) async fn try_restore_quarantine(
    quarantine_path: &Path,
    lock_path: &Path,
) -> Result<RestoreOutcome, std::io::Error> {
    // Use atomic create-or-fail (hard_link, then create_new fallback) so a fresh lock
    // installed by another owner during our quarantine isn't clobbered by `rename`.
    match tokio::fs::hard_link(quarantine_path, lock_path).await {
        Ok(()) => {
            let _ = tokio::fs::remove_file(quarantine_path).await;
            return Ok(RestoreOutcome::Restored);
        },
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let _ = tokio::fs::remove_file(quarantine_path).await;
            return Ok(RestoreOutcome::DestinationAlreadyExists);
        },
        Err(_) => {},
    }

    let bytes = match tokio::fs::read(quarantine_path).await {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = tokio::fs::remove_file(quarantine_path).await;
            return Err(error);
        },
    };
    let mut file = match tokio::fs::OpenOptions::new().write(true).create_new(true).open(lock_path).await {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let _ = tokio::fs::remove_file(quarantine_path).await;
            return Ok(RestoreOutcome::DestinationAlreadyExists);
        },
        Err(error) => {
            let _ = tokio::fs::remove_file(quarantine_path).await;
            return Err(error);
        },
    };
    if let Err(error) = file.write_all(&bytes).await {
        let _ = tokio::fs::remove_file(lock_path).await;
        let _ = tokio::fs::remove_file(quarantine_path).await;
        return Err(error);
    }
    if let Err(error) = file.sync_all().await {
        let _ = tokio::fs::remove_file(lock_path).await;
        let _ = tokio::fs::remove_file(quarantine_path).await;
        return Err(error);
    }
    let _ = tokio::fs::remove_file(quarantine_path).await;
    Ok(RestoreOutcome::Restored)
}

pub async fn try_acquire_lock(
    lock_path: &Path,
    manager_id: &str,
    instance_id: Uuid,
) -> Result<bool, std::io::Error> {
    if check_lock_file(lock_path, manager_id, instance_id, std::process::id()).await.is_conflict() {
        return Ok(false);
    }

    acquire_lock(lock_path, manager_id, instance_id).await?;
    Ok(true)
}

fn read_lock_file(lock_path: &Path) -> Result<LockFileInfo, Box<dyn std::error::Error>> {
    Ok(serde_json::from_str(&std::fs::read_to_string(lock_path)?)?)
}

fn classify_unparseable_lock(
    lock_path: &Path,
    snapshot: Vec<u8>,
) -> LockFileState {
    let stale_duration = std::time::Duration::from_secs((LOCK_TIMEOUT_MINUTES * 60) as u64);
    let mtime = match std::fs::metadata(lock_path).and_then(|metadata| metadata.modified()) {
        Ok(mtime) => mtime,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return LockFileState::Missing,
        Err(_) => {
            return LockFileState::OwnedByOtherApp(LockFileInfo {
                manager_id: "unknown".to_string(),
                instance_id: Uuid::nil(),
                acquired_at: chrono::Utc::now(),
                process_id: 0,
            });
        },
    };
    let age = std::time::SystemTime::now().duration_since(mtime).ok();
    if matches!(age, Some(age) if age >= stale_duration) {
        LockFileState::StaleUnparseable(snapshot)
    } else {
        LockFileState::OwnedByOtherApp(LockFileInfo {
            manager_id: "unknown".to_string(),
            instance_id: Uuid::nil(),
            acquired_at: chrono::Utc::now(),
            process_id: 0,
        })
    }
}

fn is_lock_stale(lock_info: &LockFileInfo) -> bool {
    chrono::Utc::now() - lock_info.acquired_at > chrono::Duration::minutes(LOCK_TIMEOUT_MINUTES)
}

#[cfg(unix)]
async fn is_process_alive(process_id: u32) -> bool {
    // SAFETY: `kill(pid, 0)` is a probe with no signal delivery. Direct syscall is required
    // because sandboxed macOS apps can't exec `/bin/kill`.
    let result = unsafe { libc::kill(process_id as libc::pid_t, 0) };
    if result == 0 {
        return true;
    }
    // EPERM = exists but we can't signal it; ESRCH = gone.
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(windows)]
async fn is_process_alive(process_id: u32) -> bool {
    tokio::task::spawn_blocking(move || {
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {process_id}")])
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).contains(&process_id.to_string()))
            .unwrap_or(false)
    })
    .await
    .unwrap_or(false)
}
