use std::path::PathBuf;

use chrono::{Duration as ChronoDuration, Utc};
use download_manager::{LockFileInfo, LockFileState, acquire_lock, release_lock_if_owned};
use uuid::Uuid;

use crate::lock_manager::{
    ReclaimExpectation, ReclaimOutcome, RestoreOutcome, classify_same_manager_lock_without_process, reclaim_stale_lock,
    try_restore_quarantine,
};

fn lock_path(temp: &tempfile::TempDir) -> PathBuf {
    temp.path().join("destination.lock")
}

#[tokio::test(flavor = "multi_thread")]
async fn acquire_lock_does_not_overwrite_other_managers_lock() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);

    acquire_lock(&path, "manager-a", Uuid::new_v4()).await?;
    let acquired_by_b = acquire_lock(&path, "manager-b", Uuid::new_v4()).await;

    assert!(acquired_by_b.is_err());

    let info: LockFileInfo = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    assert_eq!(info.manager_id, "manager-a");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn acquire_lock_reclaims_stale_locks() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);

    let stale_info = LockFileInfo {
        manager_id: "manager-a".to_string(),
        instance_id: Uuid::new_v4(),
        acquired_at: Utc::now() - ChronoDuration::hours(2),
        process_id: 999_999,
    };
    std::fs::write(&path, serde_json::to_string(&stale_info)?)?;

    acquire_lock(&path, "manager-b", Uuid::new_v4()).await?;

    let info: LockFileInfo = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    assert_eq!(info.manager_id, "manager-b");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn acquire_lock_is_idempotent_for_same_owner() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);
    let instance_id = Uuid::new_v4();

    acquire_lock(&path, "manager-a", instance_id).await?;
    acquire_lock(&path, "manager-a", instance_id).await?;

    let info: LockFileInfo = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    assert_eq!(info.manager_id, "manager-a");
    assert_eq!(info.process_id, std::process::id());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn acquire_lock_reclaims_old_unparseable_lock() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);

    std::fs::write(&path, b"this is not valid json")?;
    let two_hours_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 60 * 60);
    let times = std::fs::FileTimes::new().set_modified(two_hours_ago).set_accessed(two_hours_ago);
    let lock_handle = std::fs::OpenOptions::new().write(true).open(&path)?;
    lock_handle.set_times(times)?;
    drop(lock_handle);

    acquire_lock(&path, "manager-b", Uuid::new_v4()).await?;

    let info: LockFileInfo = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    assert_eq!(info.manager_id, "manager-b");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn acquire_lock_does_not_steal_a_recent_unparseable_lock() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);

    std::fs::write(&path, b"")?;

    let result = acquire_lock(&path, "manager-b", Uuid::new_v4()).await;

    assert!(result.is_err());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn acquire_lock_blocks_same_manager_id_with_different_instance_id() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);

    acquire_lock(&path, "manager-a", Uuid::new_v4()).await?;
    let result = acquire_lock(&path, "manager-a", Uuid::new_v4()).await;

    assert!(result.is_err());
    Ok(())
}

#[test]
fn same_manager_lock_without_process_blocks_different_recent_instance() {
    let other_instance_id = Uuid::new_v4();
    let lock_info = LockFileInfo {
        manager_id: "manager-a".to_string(),
        instance_id: other_instance_id,
        acquired_at: Utc::now(),
        process_id: 0,
    };

    let state = classify_same_manager_lock_without_process(lock_info, Uuid::new_v4());

    assert!(matches!(state, LockFileState::OwnedByOtherApp(info) if info.instance_id == other_instance_id));
}

#[test]
fn same_manager_lock_without_process_reclaims_only_after_stale() {
    let other_instance_id = Uuid::new_v4();
    let lock_info = LockFileInfo {
        manager_id: "manager-a".to_string(),
        instance_id: other_instance_id,
        acquired_at: Utc::now() - ChronoDuration::hours(2),
        process_id: 0,
    };

    let state = classify_same_manager_lock_without_process(lock_info, Uuid::new_v4());

    assert!(matches!(state, LockFileState::Stale(info) if info.instance_id == other_instance_id));
}

#[tokio::test(flavor = "multi_thread")]
async fn check_lock_file_treats_disappearing_lock_as_missing() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);

    let result = acquire_lock(&path, "self-manager", Uuid::new_v4()).await;

    assert!(result.is_ok());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn release_lock_if_owned_leaves_other_owners_lock_intact() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);

    let other_owner = LockFileInfo {
        manager_id: "other-manager".to_string(),
        instance_id: Uuid::new_v4(),
        acquired_at: Utc::now(),
        process_id: std::process::id(),
    };
    std::fs::write(&path, serde_json::to_string(&other_owner)?)?;

    let released = release_lock_if_owned(&path, "self-manager", Uuid::new_v4()).await?;

    assert!(!released);
    let info: LockFileInfo = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    assert_eq!(info.manager_id, "other-manager");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn release_lock_if_owned_removes_our_own_lock() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);
    let instance_id = Uuid::new_v4();

    acquire_lock(&path, "self-manager", instance_id).await?;
    let released = release_lock_if_owned(&path, "self-manager", instance_id).await?;

    assert!(released);
    assert!(!path.exists());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn acquire_lock_does_not_steal_live_lock_from_other_manager() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);

    let live_owner = LockFileInfo {
        manager_id: "other-manager".to_string(),
        instance_id: Uuid::new_v4(),
        acquired_at: Utc::now() - ChronoDuration::hours(2),
        process_id: std::process::id(),
    };
    std::fs::write(&path, serde_json::to_string(&live_owner)?)?;

    let result = acquire_lock(&path, "manager-b", Uuid::new_v4()).await;

    assert!(result.is_err());
    let info: LockFileInfo = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    assert_eq!(info.manager_id, "other-manager");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn reclaim_unparseable_snapshot_refuses_when_bytes_changed() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);
    std::fs::write(&path, b"different bytes than the original snapshot")?;

    let stale_snapshot = b"old garbage that was observed earlier".to_vec();
    let reclaimed = reclaim_stale_lock(&path, ReclaimExpectation::UnparseableSnapshot(stale_snapshot)).await?;

    assert!(matches!(reclaimed, ReclaimOutcome::Changed));
    let restored = std::fs::read(&path)?;
    assert_eq!(restored, b"different bytes than the original snapshot");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn reclaim_unparseable_snapshot_succeeds_when_bytes_match() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);
    let snapshot = b"identical garbage".to_vec();
    std::fs::write(&path, &snapshot)?;

    let reclaimed = reclaim_stale_lock(&path, ReclaimExpectation::UnparseableSnapshot(snapshot)).await?;

    assert_eq!(reclaimed, ReclaimOutcome::Reclaimed);
    assert!(!path.exists());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn try_restore_quarantine_does_not_overwrite_a_new_lock_at_destination() -> Result<(), Box<dyn std::error::Error>>
{
    let temp = tempfile::tempdir()?;
    let lock_target = lock_path(&temp);
    let quarantine = lock_target.with_extension("reclaim-test");
    std::fs::write(&quarantine, b"quarantined-stale-content")?;

    let new_owner = LockFileInfo {
        manager_id: "new-owner".to_string(),
        instance_id: Uuid::new_v4(),
        acquired_at: Utc::now(),
        process_id: std::process::id(),
    };
    std::fs::write(&lock_target, serde_json::to_string(&new_owner)?)?;

    let restore_outcome = try_restore_quarantine(&quarantine, &lock_target).await?;

    assert_eq!(restore_outcome, RestoreOutcome::DestinationAlreadyExists);
    assert!(!quarantine.exists());
    let intact: LockFileInfo = serde_json::from_str(&std::fs::read_to_string(&lock_target)?)?;
    assert_eq!(intact.manager_id, "new-owner");
    Ok(())
}
