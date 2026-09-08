use download_manager::{DestinationLock, LockError};
use uuid::Uuid;

#[tokio::test(flavor = "multi_thread")]
async fn lock_rules() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let destination = temp.path().join("nested").join("destination");
    let instance_id = Uuid::new_v4();
    assert_eq!(DestinationLock::foreign_owner(&destination, "manager-a", instance_id).await, None);

    let lock = DestinationLock::acquire(&destination, "manager-a", instance_id).await?;
    let owner = DestinationLock::owner(&destination).await;
    assert_eq!(owner.manager_id, "manager-a");
    assert_eq!(owner.instance_id, instance_id);
    assert_eq!(DestinationLock::foreign_owner(&destination, "manager-a", instance_id).await, None);
    assert_eq!(
        DestinationLock::foreign_owner(&destination, "manager-b", Uuid::new_v4()).await,
        Some("manager-a".to_string())
    );
    assert!(matches!(
        DestinationLock::acquire(&destination, "manager-b", Uuid::new_v4()).await,
        Err(LockError::LockedByOther { manager_id }) if manager_id == "manager-a"
    ));

    drop(lock);
    assert_eq!(DestinationLock::foreign_owner(&destination, "manager-b", Uuid::new_v4()).await, None);
    DestinationLock::acquire(&destination, "manager-b", Uuid::new_v4()).await?;
    Ok(())
}
