use std::{
    io::{Error as IoError, ErrorKind},
    path::{Path, PathBuf},
    time::Duration,
};

use kiban::{
    fs,
    process::{is_process_alive, proc_supported},
    time::SystemTime,
};
use uuid::Uuid;

use crate::{LockFileInfo, LockFileState, lock_manager::reclaim_expectation::ReclaimExpectation};

const LOCK_TIMEOUT_MINUTES: i64 = 30;

pub fn lock_path_for_destination(destination: &Path) -> PathBuf {
    PathBuf::from(format!("{}.lock", destination.display()))
}

pub async fn check_lock_file(
    lock_path: &Path,
    our_manager_id: &str,
    our_instance_id: Uuid,
    our_process_id: u32,
) -> LockFileState {
    let bytes = match fs::asyn::read(lock_path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => return LockFileState::Missing,
        Err(_) => return classify_unparseable_lock(lock_path, Vec::new()).await,
    };

    let lock_info = match serde_json::from_slice::<LockFileInfo>(&bytes) {
        Ok(lock_info) => lock_info,
        Err(_) => return classify_unparseable_lock(lock_path, bytes).await,
    };

    if lock_info.manager_id == our_manager_id {
        if proc_supported() {
            if lock_info.process_id == our_process_id {
                if lock_info.instance_id == our_instance_id {
                    return LockFileState::OwnedByUs(lock_info);
                }
                return LockFileState::OwnedByOtherApp(lock_info);
            }

            if is_process_alive(lock_info.process_id).await {
                return LockFileState::OwnedByOtherApp(lock_info);
            }
        } else {
            return classify_same_manager_lock_without_process(lock_info, our_instance_id);
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
) -> Result<(), IoError> {
    let lock_info = LockFileInfo::new(manager_id.to_string(), instance_id, kiban::process::id());
    let lock_contents = serde_json::to_string_pretty(&lock_info).map_err(IoError::other)?;

    if let Some(parent) = lock_path.parent() {
        fs::asyn::create_dir_all(parent).await?;
    }

    match write_new_lock_file(lock_path, lock_contents.as_bytes()).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            match check_lock_file(lock_path, manager_id, instance_id, kiban::process::id()).await {
                LockFileState::OwnedByUs(_) => Ok(()),
                LockFileState::OwnedBySameAppOldProcess(observed) | LockFileState::Stale(observed) => {
                    reclaim_then_retry(lock_path, manager_id, instance_id, ReclaimExpectation::Matching(observed)).await
                },
                LockFileState::StaleUnparseable(snapshot) => {
                    reclaim_then_retry(
                        lock_path,
                        manager_id,
                        instance_id,
                        ReclaimExpectation::UnparseableSnapshot(snapshot),
                    )
                    .await
                },
                LockFileState::OwnedByOtherApp(info) => {
                    Err(IoError::new(ErrorKind::AlreadyExists, format!("destination locked by {}", info.manager_id)))
                },
                LockFileState::Missing => Box::pin(acquire_lock(lock_path, manager_id, instance_id)).await,
            }
        },
        Err(error) => Err(error),
    }
}

pub async fn release_lock_if_owned(
    lock_path: &Path,
    manager_id: &str,
    instance_id: Uuid,
) -> Result<bool, IoError> {
    let observed = match check_lock_file(lock_path, manager_id, instance_id, kiban::process::id()).await {
        LockFileState::OwnedByUs(info) => info,
        _ => return Ok(false),
    };
    let quarantine_path = lock_path.with_extension(format!("release-{}", Uuid::new_v4()));
    match fs::asyn::rename(lock_path, &quarantine_path).await {
        Ok(()) => {},
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    }
    let parsed = read_lock_file(&quarantine_path).await.ok();
    if matches!(parsed.as_ref(), Some(actual) if actual == &observed) {
        let _ = fs::asyn::remove_file(&quarantine_path).await;
        return Ok(true);
    }
    try_restore_quarantine(&quarantine_path, lock_path).await?;
    Ok(false)
}

async fn reclaim_then_retry(
    lock_path: &Path,
    manager_id: &str,
    instance_id: Uuid,
    expectation: ReclaimExpectation,
) -> Result<(), IoError> {
    if reclaim_stale_lock(lock_path, expectation).await? {
        Box::pin(acquire_lock(lock_path, manager_id, instance_id)).await
    } else {
        Err(IoError::new(ErrorKind::AlreadyExists, "destination lock changed while reclaiming stale lock"))
    }
}

async fn reclaim_stale_lock(
    lock_path: &Path,
    expectation: ReclaimExpectation,
) -> Result<bool, IoError> {
    let quarantine_path = lock_path.with_extension(format!("reclaim-{}", Uuid::new_v4()));
    match fs::asyn::rename(lock_path, &quarantine_path).await {
        Ok(()) => {},
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(true),
        Err(error) => return Err(error),
    }
    let confirmed = match &expectation {
        ReclaimExpectation::Matching(observed) => {
            matches!(read_lock_file(&quarantine_path).await.ok(), Some(actual) if &actual == observed)
        },
        ReclaimExpectation::UnparseableSnapshot(snapshot) => {
            matches!(fs::asyn::read(&quarantine_path).await.ok(), Some(bytes) if &bytes == snapshot)
        },
    };
    if confirmed {
        let _ = fs::asyn::remove_file(&quarantine_path).await;
        return Ok(true);
    }
    try_restore_quarantine(&quarantine_path, lock_path).await?;
    Ok(false)
}

async fn try_restore_quarantine(
    quarantine_path: &Path,
    lock_path: &Path,
) -> Result<(), IoError> {
    match fs::asyn::hard_link(quarantine_path, lock_path).await {
        Ok(()) => {
            let _ = fs::asyn::remove_file(quarantine_path).await;
            return Ok(());
        },
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            let _ = fs::asyn::remove_file(quarantine_path).await;
            return Ok(());
        },
        Err(_) => {},
    }

    let bytes = match fs::asyn::read(quarantine_path).await {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = fs::asyn::remove_file(quarantine_path).await;
            return Err(error);
        },
    };

    if matches!(fs::asyn::try_exists(lock_path).await, Ok(true)) {
        let _ = fs::asyn::remove_file(quarantine_path).await;
        return Ok(());
    }

    if let Err(error) = fs::asyn::write_with_sync_all(&lock_path, &bytes).await {
        let _ = fs::asyn::remove_file(quarantine_path).await;
        if error.kind() == ErrorKind::AlreadyExists {
            return Ok(());
        }
        let _ = fs::asyn::remove_file(lock_path).await;
        return Err(error);
    }
    let _ = fs::asyn::remove_file(quarantine_path).await;
    Ok(())
}

async fn write_new_lock_file(
    lock_path: &Path,
    lock_contents: &[u8],
) -> Result<(), IoError> {
    if !proc_supported() {
        return write_new_lock_file_direct(lock_path, lock_contents).await;
    }

    let temporary_lock_path = lock_path.with_extension(format!("lock-tmp-{}", Uuid::new_v4()));
    if let Err(error) = fs::asyn::write_with_sync_all(&temporary_lock_path, &lock_contents).await {
        let _ = fs::asyn::remove_file(&temporary_lock_path).await;
        return Err(error);
    }

    match fs::asyn::hard_link(&temporary_lock_path, lock_path).await {
        Ok(()) => {
            let _ = fs::asyn::remove_file(&temporary_lock_path).await;
            Ok(())
        },
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            let _ = fs::asyn::remove_file(&temporary_lock_path).await;
            Err(error)
        },
        Err(_) => {
            let result = write_new_lock_file_direct(lock_path, lock_contents).await;
            let _ = fs::asyn::remove_file(&temporary_lock_path).await;
            result
        },
    }
}

async fn write_new_lock_file_direct(
    lock_path: &Path,
    lock_contents: &[u8],
) -> Result<(), IoError> {
    if let Err(error) = fs::asyn::write_with_sync_all(&lock_path, &lock_contents).await {
        if error.kind() != ErrorKind::AlreadyExists {
            let _ = fs::asyn::remove_file(&lock_path).await;
        }
        return Err(error);
    }
    Ok(())
}

async fn read_lock_file(lock_path: &Path) -> Result<LockFileInfo, Box<dyn std::error::Error>> {
    let file_content = fs::asyn::read_to_string(&lock_path).await?;
    Ok(serde_json::from_str(&file_content)?)
}

fn classify_same_manager_lock_without_process(
    lock_info: LockFileInfo,
    our_instance_id: Uuid,
) -> LockFileState {
    if lock_info.instance_id == our_instance_id {
        return LockFileState::OwnedByUs(lock_info);
    }

    if is_lock_stale(&lock_info) {
        LockFileState::Stale(lock_info)
    } else {
        LockFileState::OwnedByOtherApp(lock_info)
    }
}

async fn classify_unparseable_lock(
    lock_path: &Path,
    snapshot: Vec<u8>,
) -> LockFileState {
    let stale_duration = Duration::from_secs((LOCK_TIMEOUT_MINUTES * 60) as u64);
    let mtime = match fs::asyn::file_modified(lock_path).await {
        Ok(mtime) => mtime,
        Err(error) if error.kind() == ErrorKind::NotFound => return LockFileState::Missing,
        Err(_) => return LockFileState::OwnedByOtherApp(unknown_lock_info()),
    };
    let age = SystemTime::now().duration_since(mtime).ok();
    if matches!(age, Some(age) if age >= stale_duration) {
        LockFileState::StaleUnparseable(snapshot)
    } else {
        LockFileState::OwnedByOtherApp(unknown_lock_info())
    }
}

fn unknown_lock_info() -> LockFileInfo {
    LockFileInfo {
        manager_id: "unknown".to_string(),
        instance_id: Uuid::nil(),
        acquired_at: chrono::Utc::now(),
        process_id: 0,
    }
}

fn is_lock_stale(lock_info: &LockFileInfo) -> bool {
    chrono::Utc::now() - lock_info.acquired_at > chrono::Duration::minutes(LOCK_TIMEOUT_MINUTES)
}
