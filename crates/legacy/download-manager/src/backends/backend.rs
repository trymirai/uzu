use std::{
    io::ErrorKind,
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
            && !fs::asyn::is_file(CrcReceipt::path_for(&config.destination)).await
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
            }) => return Ok((self.observe(config, Some(manager_id)).await.0, None)),
            Err(error) => return Err(error.into()),
        };
        let (lifecycle, cleanup) = self.observe(config, None).await;
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

    async fn observe(
        &self,
        config: &DownloadConfig,
        foreign_owner: Option<String>,
    ) -> (Lifecycle, Vec<PathBuf>) {
        let receipt_path = CrcReceipt::path_for(&config.destination);
        let receipt_exists = fs::asyn::is_file(&receipt_path).await;
        let resume_bytes = if fs::asyn::is_file(&config.resume_artifact_path).await {
            Some(self.read_resume_progress(&config.resume_artifact_path).await)
        } else {
            None
        };
        let destination_size = if fs::asyn::is_file(&config.destination).await {
            fs::asyn::file_length(&config.destination).await.ok()
        } else {
            None
        };
        let mut cleanup = Vec::new();
        let downloaded = match destination_size {
            Some(size) if self.verify(config, size).await.is_ok() => {
                cleanup.extend(resume_bytes.map(|_| config.resume_artifact_path.clone()));
                Some(size)
            },
            Some(_) => {
                cleanup.push(config.destination.clone());
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

    async fn verify_download(
        &self,
        config: &DownloadConfig,
    ) -> Result<u64, String> {
        let size = fs::asyn::file_length(&config.destination).await.map_err(|error| error.to_string())?;
        self.verify(config, size).await?;
        Ok(size)
    }

    async fn verify(
        &self,
        config: &DownloadConfig,
        size: u64,
    ) -> Result<(), String> {
        if let Some(expected_bytes) = config.expected_bytes
            && expected_bytes != size
        {
            return Err(format!("downloaded file is {size} bytes but registry declared {expected_bytes}"));
        }
        let Some(crc) = &config.expected_crc32c else {
            return Ok(());
        };
        if crc.cached_matches(&config.destination).await {
            return Ok(());
        }
        match crc.verify(&config.destination).await {
            Ok(true) => {
                let _ = crc.save_receipt(&config.destination).await;
                Ok(())
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
        for path in [&config.resume_artifact_path, &config.destination, &CrcReceipt::path_for(&config.destination)] {
            let _ = fs::asyn::remove_file(path).await;
        }
    }
}
