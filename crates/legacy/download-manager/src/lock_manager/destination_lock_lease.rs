use std::path::{Path, PathBuf};

use uuid::Uuid;

use crate::lock_manager::{acquire_lock, lock_path_for_destination, release_lock_if_owned};

#[derive(Debug)]
pub struct DestinationLockLease {
    lock_path: PathBuf,
    manager_id: String,
    instance_id: Uuid,
}

impl DestinationLockLease {
    pub async fn acquire_for_destination(
        destination_path: &Path,
        manager_id: &str,
        instance_id: Uuid,
    ) -> Result<Self, std::io::Error> {
        Self::acquire(&lock_path_for_destination(destination_path), manager_id, instance_id).await
    }

    pub async fn acquire(
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

    pub async fn release(self) -> Result<bool, std::io::Error> {
        release_lock_if_owned(&self.lock_path, &self.manager_id, self.instance_id).await
    }
}
