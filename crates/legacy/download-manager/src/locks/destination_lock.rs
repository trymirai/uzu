use std::path::{Path, PathBuf};

use kiban::fs::{self, FileLock};
use uuid::Uuid;

use crate::locks::{LockError, LockOwner};

pub struct DestinationLock {
    _lock: FileLock,
}

impl DestinationLock {
    pub fn path_for(destination: &Path) -> PathBuf {
        PathBuf::from(format!("{}.lock", destination.display()))
    }

    pub async fn acquire(
        lock_path: &Path,
        manager_id: &str,
        instance_id: Uuid,
    ) -> Result<Self, LockError> {
        let Some(lock) = FileLock::try_acquire(lock_path).await? else {
            return Err(LockError::LockedByOther {
                manager_id: Self::owner(lock_path).await.manager_id,
            });
        };
        let owner = LockOwner {
            manager_id: manager_id.to_string(),
            instance_id,
        };
        lock.write(&serde_json::to_vec(&owner).map_err(std::io::Error::other)?).await?;
        Ok(Self {
            _lock: lock,
        })
    }

    pub async fn foreign_owner(
        lock_path: &Path,
        manager_id: &str,
        instance_id: Uuid,
    ) -> Option<String> {
        if !fs::asyn::try_exists(lock_path).await.unwrap_or(false) {
            return None;
        }
        match FileLock::try_acquire(lock_path).await {
            Ok(None) => {
                let owner = Self::owner(lock_path).await;
                (owner.manager_id != manager_id || owner.instance_id != instance_id).then_some(owner.manager_id)
            },
            Ok(Some(_)) | Err(_) => None,
        }
    }

    pub async fn owner(lock_path: &Path) -> LockOwner {
        fs::asyn::read_to_string(lock_path)
            .await
            .ok()
            .and_then(|contents| serde_json::from_str(&contents).ok())
            .unwrap_or_else(|| LockOwner {
                manager_id: "unknown".to_string(),
                instance_id: Uuid::nil(),
            })
    }
}
