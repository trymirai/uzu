use std::path::PathBuf;

use chrono::{Duration as ChronoDuration, Utc};
use download_manager::{LockFileInfo, acquire_lock, release_lock_if_owned};
use uuid::Uuid;

use crate::lock_manager::{
    ReclaimExpectation, ReclaimOutcome, RestoreOutcome, reclaim_stale_lock, try_restore_quarantine,
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

    assert!(
        acquired_by_b.is_err(),
        "acquire_lock must not silently overwrite an existing lock from another manager",
    );

    let info: LockFileInfo = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    assert_eq!(info.manager_id, "manager-a", "manager-a's lock must remain intact");
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
    assert_eq!(info.manager_id, "manager-b", "stale lock must be reclaimed");
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
    assert_eq!(
        info.manager_id, "manager-b",
        "an old garbage lock file must be reclaimable instead of looping forever",
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn acquire_lock_does_not_steal_a_recent_unparseable_lock() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);

    std::fs::write(&path, b"")?;

    let result = acquire_lock(&path, "manager-b", Uuid::new_v4()).await;

    assert!(
        result.is_err(),
        "acquire_lock must treat a freshly-created unparseable lock as a conflict (another caller may be mid-acquire)",
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn acquire_lock_blocks_same_manager_id_with_different_instance_id() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);

    acquire_lock(&path, "manager-a", Uuid::new_v4()).await?;
    let result = acquire_lock(&path, "manager-a", Uuid::new_v4()).await;

    assert!(
        result.is_err(),
        "two manager instances in the same process with different instance_ids must not share a destination lock",
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn check_lock_file_treats_disappearing_lock_as_missing() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);

    let result = acquire_lock(&path, "self-manager", Uuid::new_v4()).await;

    assert!(result.is_ok(), "acquire_lock must succeed when no lock file exists, got {result:?}");
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

    assert!(!released, "release must refuse to delete a lock owned by a different manager");
    let info: LockFileInfo = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    assert_eq!(info.manager_id, "other-manager", "the foreign owner's lock must remain on disk");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn release_lock_if_owned_removes_our_own_lock() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);
    let instance_id = Uuid::new_v4();

    acquire_lock(&path, "self-manager", instance_id).await?;
    let released = release_lock_if_owned(&path, "self-manager", instance_id).await?;

    assert!(released, "release must succeed when we are the lock owner");
    assert!(!path.exists(), "the lock file must be gone after a successful release");
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

    assert!(
        result.is_err(),
        "a different manager's lock owned by a live process must never be reclaimed, even after the staleness timeout",
    );
    let info: LockFileInfo = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    assert_eq!(info.manager_id, "other-manager", "live owner's lock must remain intact");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn reclaim_unparseable_snapshot_refuses_when_bytes_changed() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);
    std::fs::write(&path, b"different bytes than the original snapshot")?;

    let stale_snapshot = b"old garbage that was observed earlier".to_vec();
    let reclaimed = reclaim_stale_lock(&path, ReclaimExpectation::UnparseableSnapshot(stale_snapshot)).await?;

    assert!(
        matches!(reclaimed, ReclaimOutcome::Changed),
        "unparseable reclaim must refuse to delete bytes that no longer match the original snapshot",
    );
    let restored = std::fs::read(&path)?;
    assert_eq!(
        restored, b"different bytes than the original snapshot",
        "the on-disk file must be restored after a snapshot mismatch",
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn reclaim_unparseable_snapshot_succeeds_when_bytes_match() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = lock_path(&temp);
    let snapshot = b"identical garbage".to_vec();
    std::fs::write(&path, &snapshot)?;

    let reclaimed = reclaim_stale_lock(&path, ReclaimExpectation::UnparseableSnapshot(snapshot)).await?;

    assert_eq!(
        reclaimed,
        ReclaimOutcome::Reclaimed,
        "unparseable reclaim must succeed when bytes still match the snapshot",
    );
    assert!(!path.exists(), "the lock file must be gone after a confirmed reclaim");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn try_restore_quarantine_does_not_overwrite_a_new_lock_at_destination()
-> Result<(), Box<dyn std::error::Error>> {
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
    assert!(
        !quarantine.exists(),
        "the quarantine path must be cleaned up regardless of whether the restore could be applied",
    );
    let intact: LockFileInfo = serde_json::from_str(&std::fs::read_to_string(&lock_target)?)?;
    assert_eq!(
        intact.manager_id, "new-owner",
        "restore must not overwrite a fresh lock that another owner created in the race window",
    );
    Ok(())
}
