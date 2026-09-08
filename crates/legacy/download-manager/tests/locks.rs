use download_manager::{DestinationLock, LockError, LockOwner};
use uuid::Uuid;

#[tokio::test(flavor = "multi_thread")]
async fn lock_rules() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let destination = temp.path().join("nested").join("destination");
    let owner_a = LockOwner {
        manager_id: "manager-a".to_string(),
        instance_id: Uuid::new_v4(),
    };
    let owner_b = LockOwner {
        manager_id: "manager-b".to_string(),
        instance_id: Uuid::new_v4(),
    };
    assert_eq!(DestinationLock::foreign_owner(&destination, &owner_a).await, None);

    let lock = DestinationLock::acquire(&destination, &owner_a).await?;
    assert_eq!(DestinationLock::owner(&destination).await, owner_a);
    assert_eq!(DestinationLock::foreign_owner(&destination, &owner_a).await, None);
    assert_eq!(DestinationLock::foreign_owner(&destination, &owner_b).await, Some("manager-a".to_string()));
    assert!(matches!(
        DestinationLock::acquire(&destination, &owner_b).await,
        Err(LockError::LockedByOther { manager_id }) if manager_id == "manager-a"
    ));

    drop(lock);
    assert_eq!(DestinationLock::foreign_owner(&destination, &owner_b).await, None);
    DestinationLock::acquire(&destination, &owner_b).await?;
    Ok(())
}
