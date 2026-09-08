use download_manager::{DestinationLock, LockError};
use uuid::Uuid;

#[tokio::test(flavor = "multi_thread")]
async fn lock_rules() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("nested").join("destination.lock");
    let instance_id = Uuid::new_v4();
    assert_eq!(DestinationLock::foreign_owner(&path, "manager-a", instance_id).await, None);

    let lock = DestinationLock::acquire(&path, "manager-a", instance_id).await?;
    let owner = DestinationLock::owner(&path).await;
    assert_eq!(owner.manager_id, "manager-a");
    assert_eq!(owner.instance_id, instance_id);
    assert_eq!(DestinationLock::foreign_owner(&path, "manager-a", instance_id).await, None);
    assert_eq!(DestinationLock::foreign_owner(&path, "manager-b", Uuid::new_v4()).await, Some("manager-a".to_string()));
    assert!(matches!(
        DestinationLock::acquire(&path, "manager-b", Uuid::new_v4()).await,
        Err(LockError::LockedByOther { manager_id }) if manager_id == "manager-a"
    ));

    drop(lock);
    assert_eq!(DestinationLock::foreign_owner(&path, "manager-b", Uuid::new_v4()).await, None);
    DestinationLock::acquire(&path, "manager-b", Uuid::new_v4()).await?;
    Ok(())
}
