use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use kiban::fs;

use crate::{
    DownloadError,
    backends::{ActiveTask, BackendEventSender, DownloadGeneration},
    crc_receipt::CrcReceipt,
    file_download::{DownloadConfig, Lifecycle},
    locks::{DestinationLock, LockError},
};

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;

    fn resume_artifact_extension(&self) -> &'static str;

    async fn start(
        &self,
        config: Arc<DownloadConfig>,
        generation: DownloadGeneration,
        events: BackendEventSender,
    ) -> Result<Box<dyn ActiveTask>, DownloadError>;

    async fn read_resume_progress(
        &self,
        resume_artifact_path: &Path,
    ) -> u64;

    async fn has_pending_task(
        &self,
        config: &DownloadConfig,
    ) -> Result<bool, DownloadError>;

    async fn attach_pending_task(
        &self,
        config: Arc<DownloadConfig>,
        generation: DownloadGeneration,
        events: BackendEventSender,
    ) -> Result<Option<Box<dyn ActiveTask>>, DownloadError>;

    fn resume_artifact_path(
        &self,
        destination: &Path,
    ) -> PathBuf {
        PathBuf::from(format!("{}.{}", destination.display(), self.resume_artifact_extension()))
    }

    async fn lock(
        &self,
        config: &DownloadConfig,
    ) -> Result<DestinationLock, LockError> {
        DestinationLock::acquire(&config.destination, &config.manager_id, config.manager_instance_id).await
    }

    async fn reconcile(
        &self,
        config: &DownloadConfig,
    ) -> Result<(Lifecycle, Option<DestinationLock>), DownloadError> {
        let untouched = !fs::asyn::is_file(&config.destination).await
            && !fs::asyn::is_file(&config.resume_artifact_path).await
            && !CrcReceipt::exists(&config.destination).await
            && DestinationLock::foreign_owner(&config.destination, &config.manager_id, config.manager_instance_id)
                .await
                .is_none();
        let pending_task = self.has_pending_task(config).await?;
        if untouched && !pending_task {
            return Ok((Lifecycle::NotDownloaded, None));
        }
        let lock = match self.lock(config).await {
            Ok(lock) => lock,
            Err(LockError::LockedByOther {
                manager_id,
            }) => return Ok((self.observe(config, Some(manager_id)).await, None)),
            Err(error) => return Err(error.into()),
        };
        let lifecycle = self.observe(config, None).await;
        let attach = pending_task && !matches!(lifecycle, Lifecycle::Downloaded { .. });
        Ok((lifecycle, attach.then_some(lock)))
    }

    async fn observe(
        &self,
        config: &DownloadConfig,
        foreign_owner: Option<String>,
    ) -> Lifecycle {
        let resume_bytes = if fs::asyn::is_file(&config.resume_artifact_path).await {
            Some(self.read_resume_progress(&config.resume_artifact_path).await)
        } else {
            None
        };
        let downloaded = if fs::asyn::is_file(&config.destination).await {
            self.verify(config).await.ok()
        } else {
            None
        };
        if foreign_owner.is_none() {
            if downloaded.is_some() {
                self.remove_resume_artifact(config).await;
            } else {
                let _ = fs::asyn::remove_file(&config.destination).await;
                CrcReceipt::remove(&config.destination).await;
            }
        }
        match (downloaded, resume_bytes, foreign_owner) {
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
        }
    }

    async fn verify(
        &self,
        config: &DownloadConfig,
    ) -> Result<u64, String> {
        let size = fs::asyn::file_length(&config.destination).await.map_err(|error| error.to_string())?;
        if let Some(expected_bytes) = config.expected_bytes
            && expected_bytes != size
        {
            return Err(format!("downloaded file is {size} bytes but registry declared {expected_bytes}"));
        }
        let Some(crc) = &config.expected_crc32c else {
            return Ok(size);
        };
        if CrcReceipt::matches(&config.destination, crc).await {
            return Ok(size);
        }
        match crc.verify(&config.destination).await {
            Ok(true) => {
                let _ = CrcReceipt::save(&config.destination, crc).await;
                Ok(size)
            },
            Ok(false) => Err("CRC verification failed".to_string()),
            Err(error) => Err(format!("CRC verification error: {error}")),
        }
    }

    async fn remove_resume_artifact(
        &self,
        config: &DownloadConfig,
    ) {
        let _ = fs::asyn::remove_file(&config.resume_artifact_path).await;
    }

    async fn remove_files(
        &self,
        config: &DownloadConfig,
    ) {
        self.remove_resume_artifact(config).await;
        let _ = fs::asyn::remove_file(&config.destination).await;
        CrcReceipt::remove(&config.destination).await;
    }
}
