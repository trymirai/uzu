use std::{io::ErrorKind, path::PathBuf};

use kiban::fs;
use uuid::Uuid;

use crate::{
    Crc32c, DownloadError, DownloadId,
    backends::Backend,
    crc_receipt::CrcReceipt,
    file_download::Lifecycle,
    locks::{DestinationLock, LockError},
};

pub struct DownloadConfig {
    pub download_id: DownloadId,
    pub source_url: String,
    pub destination: PathBuf,
    pub resume_artifact_path: PathBuf,
    pub expected_crc32c: Option<Crc32c>,
    pub expected_bytes: Option<u64>,
    pub manager_id: String,
    pub manager_instance_id: Uuid,
}

impl DownloadConfig {
    pub fn lock_path(&self) -> PathBuf {
        DestinationLock::path_for(&self.destination)
    }

    pub async fn lock(&self) -> Result<DestinationLock, LockError> {
        DestinationLock::acquire(&self.lock_path(), &self.manager_id, self.manager_instance_id).await
    }

    pub async fn foreign_owner(&self) -> Option<String> {
        DestinationLock::foreign_owner(&self.lock_path(), &self.manager_id, self.manager_instance_id).await
    }

    pub async fn reconcile(
        &self,
        backend: &dyn Backend,
    ) -> Result<(Lifecycle, Option<DestinationLock>), DownloadError> {
        let untouched = !fs::asyn::is_file(&self.destination).await
            && !fs::asyn::is_file(&self.resume_artifact_path).await
            && !fs::asyn::is_file(CrcReceipt::path_for(&self.destination)).await
            && !fs::asyn::is_file(self.lock_path()).await;
        let pending_task = backend.has_pending_task(self).await?;
        if untouched && !pending_task {
            return Ok((Lifecycle::NotDownloaded, None));
        }
        let lock = match self.lock().await {
            Ok(lock) => lock,
            Err(LockError::LockedByOther {
                manager_id,
            }) => return Ok((self.observe(backend, Some(manager_id)).await.0, None)),
            Err(error) => return Err(error.into()),
        };
        let (lifecycle, cleanup) = self.observe(backend, None).await;
        for path in cleanup {
            if let Err(error) = fs::asyn::remove_file(&path).await
                && error.kind() != ErrorKind::NotFound
            {
                return Err(error.into());
            }
        }
        let attach = pending_task && !matches!(lifecycle, Lifecycle::Downloaded { .. });
        Ok((lifecycle, attach.then_some(lock)))
    }

    pub async fn verify_download(&self) -> Result<u64, String> {
        let size = fs::asyn::file_length(&self.destination).await.map_err(|error| error.to_string())?;
        self.verify(size).await?;
        Ok(size)
    }

    pub async fn remove_resume_artifact(&self) {
        let _ = fs::asyn::remove_file(&self.resume_artifact_path).await;
    }

    pub async fn remove_files(&self) {
        for path in [&self.resume_artifact_path, &self.destination, &CrcReceipt::path_for(&self.destination)] {
            let _ = fs::asyn::remove_file(path).await;
        }
    }

    async fn observe(
        &self,
        backend: &dyn Backend,
        foreign_owner: Option<String>,
    ) -> (Lifecycle, Vec<PathBuf>) {
        let receipt_path = CrcReceipt::path_for(&self.destination);
        let receipt_exists = fs::asyn::is_file(&receipt_path).await;
        let resume_bytes = if fs::asyn::is_file(&self.resume_artifact_path).await {
            Some(backend.read_resume_progress(&self.resume_artifact_path).await)
        } else {
            None
        };
        let destination_size = if fs::asyn::is_file(&self.destination).await {
            fs::asyn::file_length(&self.destination).await.ok()
        } else {
            None
        };
        let mut cleanup = Vec::new();
        let downloaded = match destination_size {
            Some(size) if self.verify(size).await.is_ok() => {
                cleanup.extend(resume_bytes.map(|_| self.resume_artifact_path.clone()));
                Some(size)
            },
            Some(_) => {
                cleanup.push(self.destination.clone());
                cleanup.extend(receipt_exists.then_some(receipt_path));
                None
            },
            None => {
                cleanup.extend(receipt_exists.then_some(receipt_path));
                None
            },
        };
        let lifecycle = match (downloaded, resume_bytes, foreign_owner) {
            (Some(total_bytes), _, _) => Lifecycle::Downloaded {
                total_bytes,
            },
            (None, downloaded_bytes, Some(manager_id)) => Lifecycle::Locked {
                manager_id,
                downloaded_bytes: downloaded_bytes.unwrap_or(0),
            },
            (None, Some(downloaded_bytes), None) => Lifecycle::Paused {
                downloaded_bytes,
            },
            (None, None, None) => Lifecycle::NotDownloaded,
        };
        (lifecycle, cleanup)
    }

    async fn verify(
        &self,
        size: u64,
    ) -> Result<(), String> {
        if let Some(expected_bytes) = self.expected_bytes
            && expected_bytes != size
        {
            return Err(format!("downloaded file is {size} bytes but registry declared {expected_bytes}"));
        }
        let Some(crc) = &self.expected_crc32c else {
            return Ok(());
        };
        if crc.cached_matches(&self.destination).await {
            return Ok(());
        }
        match crc.verify(&self.destination).await {
            Ok(true) => {
                let _ = crc.save_receipt(&self.destination).await;
                Ok(())
            },
            Ok(false) => Err("CRC verification failed".to_string()),
            Err(error) => Err(format!("CRC verification error: {error}")),
        }
    }
}
