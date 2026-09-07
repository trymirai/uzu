use std::path::Path;

use chrono::{Duration as ChronoDuration, Utc};
use download_manager::{LockFileInfo, LockFileState, acquire_lock, check_lock_file, release_lock_if_owned};
use uuid::Uuid;

fn read(path: &Path) -> Result<LockFileInfo, Box<dyn std::error::Error>> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

fn write(
    path: &Path,
    manager_id: &str,
    acquired_at: chrono::DateTime<Utc>,
    process_id: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let info = LockFileInfo {
        manager_id: manager_id.to_string(),
        instance_id: Uuid::new_v4(),
        acquired_at,
        process_id,
    };
    std::fs::write(path, serde_json::to_string(&info)?)?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn acquire_lock_ownership_rules() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("destination.lock");
    let instance_id = Uuid::new_v4();

    acquire_lock(&path, "manager-a", instance_id).await?;
    acquire_lock(&path, "manager-a", instance_id).await?;
    assert_eq!(read(&path)?.process_id, std::process::id());
    assert!(acquire_lock(&path, "manager-a", Uuid::new_v4()).await.is_err());
    assert!(acquire_lock(&path, "manager-b", Uuid::new_v4()).await.is_err());
    assert_eq!(read(&path)?.manager_id, "manager-a");

    write(&path, "other-manager", Utc::now() - ChronoDuration::hours(2), std::process::id())?;
    assert!(acquire_lock(&path, "manager-b", Uuid::new_v4()).await.is_err());
    assert_eq!(read(&path)?.manager_id, "other-manager");

    write(&path, "other-manager", Utc::now() - ChronoDuration::hours(2), 999_999)?;
    acquire_lock(&path, "manager-b", Uuid::new_v4()).await?;
    assert_eq!(read(&path)?.manager_id, "manager-b");

    std::fs::write(&path, b"this is not valid json")?;
    let two_hours_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 60 * 60);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)?
        .set_times(std::fs::FileTimes::new().set_modified(two_hours_ago).set_accessed(two_hours_ago))?;
    acquire_lock(&path, "manager-c", Uuid::new_v4()).await?;
    assert_eq!(read(&path)?.manager_id, "manager-c");

    std::fs::write(&path, b"")?;
    assert!(acquire_lock(&path, "manager-d", Uuid::new_v4()).await.is_err());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn release_lock_rules() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("destination.lock");
    let instance_id = Uuid::new_v4();

    assert!(matches!(
        check_lock_file(&path, "self-manager", instance_id, std::process::id()).await,
        LockFileState::Missing
    ));

    write(&path, "other-manager", Utc::now(), std::process::id())?;
    assert!(!release_lock_if_owned(&path, "self-manager", instance_id).await?);
    assert_eq!(read(&path)?.manager_id, "other-manager");

    std::fs::remove_file(&path)?;
    acquire_lock(&path, "self-manager", instance_id).await?;
    assert!(release_lock_if_owned(&path, "self-manager", instance_id).await?);
    assert!(!path.exists());
    Ok(())
}
