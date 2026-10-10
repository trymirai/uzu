use std::path::{Path, PathBuf};

use kiban::fs::{self, FileLock};
use uuid::Uuid;

use crate::locks::{LockError, LockOwner};

pub struct DestinationLock {
    path: PathBuf,
    _lock: FileLock,
}

impl DestinationLock {
    pub(crate) async fn exists(destination: &Path) -> bool {
        fs::asyn::try_exists(Self::path_for(destination)).await.unwrap_or(false)
    }

    pub async fn acquire(
        destination: &Path,
        owner: &LockOwner,
    ) -> Result<Self, LockError> {
        let path = Self::path_for(destination);
        let Some(lock) = FileLock::try_acquire(&path).await? else {
            return Err(LockError::LockedByOther {
                manager_id: Self::owner(destination).await.manager_id,
            });
        };
        lock.write(&serde_json::to_vec(owner).map_err(std::io::Error::other)?).await?;
        Ok(Self {
            path,
            _lock: lock,
        })
    }

    pub async fn foreign_owner(
        destination: &Path,
        owner: &LockOwner,
    ) -> Option<String> {
        let path = Self::path_for(destination);
        if !fs::asyn::try_exists(&path).await.unwrap_or(false) {
            return None;
        }
        match FileLock::try_acquire(&path).await {
            Ok(None) => {
                let holder = Self::owner(destination).await;
                (holder != *owner).then_some(holder.manager_id)
            },
            Ok(Some(_)) | Err(_) => None,
        }
    }

    pub async fn held_within(directory: &Path) -> bool {
        let mut directories = vec![directory.to_path_buf()];
        while let Some(directory) = directories.pop() {
            let Ok(paths) = fs::asyn::read_dir(&directory).await else {
                return true;
            };
            for path in paths {
                // Cleanup must not follow links outside the checkpoint or infer
                // that unreadable contents are safe to delete.
                #[cfg(not(target_family = "wasm"))]
                if !tokio::fs::symlink_metadata(&path).await.is_ok_and(|metadata| !metadata.is_symlink()) {
                    return true;
                }
                if !fs::asyn::is_file(&path).await {
                    directories.push(path);
                } else if path.extension().is_some_and(|extension| extension == "lock")
                    && !matches!(FileLock::try_acquire(&path).await, Ok(Some(_)))
                {
                    return true;
                }
            }
        }
        false
    }

    pub async fn owner(destination: &Path) -> LockOwner {
        fs::asyn::read_to_string(Self::path_for(destination))
            .await
            .ok()
            .and_then(|contents| serde_json::from_str(&contents).ok())
            .unwrap_or_else(|| LockOwner {
                manager_id: "unknown".to_string(),
                instance_id: Uuid::nil(),
            })
    }

    fn path_for(destination: &Path) -> PathBuf {
        PathBuf::from(format!("{}.lock", destination.display()))
    }
}

impl Drop for DestinationLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
