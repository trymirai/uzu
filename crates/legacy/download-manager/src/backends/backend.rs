use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use kiban::fs;

use crate::{
    DownloadPhase, DownloadState,
    backends::{ActiveTask, BackendError, BackendEventSender, DownloadGeneration, VerifyError},
    crc_receipt::CrcReceipt,
    file_download::DownloadConfig,
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
    ) -> Result<Box<dyn ActiveTask>, BackendError>;

    async fn read_resume_progress(
        &self,
        resume_artifact_path: &Path,
    ) -> u64;

    async fn has_pending_task(
        &self,
        config: &DownloadConfig,
    ) -> Result<bool, BackendError>;

    async fn attach_pending_task(
        &self,
        config: Arc<DownloadConfig>,
        generation: DownloadGeneration,
        events: BackendEventSender,
    ) -> Result<Option<Box<dyn ActiveTask>>, BackendError>;

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
    ) -> Result<(DownloadState, Option<DestinationLock>), BackendError> {
        let untouched = !fs::asyn::is_file(&config.destination).await
            && !fs::asyn::is_file(&config.resume_artifact_path).await
            && !CrcReceipt::exists(&config.destination).await
            && DestinationLock::foreign_owner(&config.destination, &config.manager_id, config.manager_instance_id)
                .await
                .is_none();
        let pending_task = self.has_pending_task(config).await?;
        if untouched && !pending_task {
            return Ok((DownloadState::new(config, DownloadPhase::NotDownloaded {}, 0, None), None));
        }
        let lock = match self.lock(config).await {
            Ok(lock) => lock,
            Err(LockError::LockedByOther {
                manager_id,
            }) => return Ok((self.observe(config, Some(manager_id)).await, None)),
            Err(error) => return Err(error.into()),
        };
        let state = self.observe(config, None).await;
        let attach = pending_task && state.phase != (DownloadPhase::Downloaded {});
        Ok((state, attach.then_some(lock)))
    }

    async fn observe(
        &self,
        config: &DownloadConfig,
        foreign_owner: Option<String>,
    ) -> DownloadState {
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
                let _ = fs::asyn::remove_file(&config.resume_artifact_path).await;
            } else {
                let _ = fs::asyn::remove_file(&config.destination).await;
                CrcReceipt::remove(&config.destination).await;
            }
        }
        let (phase, downloaded_bytes, total_bytes) = match (downloaded, resume_bytes, foreign_owner) {
            (Some(size), _, _) => (DownloadPhase::Downloaded {}, size, Some(size)),
            (None, downloaded_bytes, Some(manager_id)) => (
                DownloadPhase::Locked {
                    manager_id,
                },
                downloaded_bytes.unwrap_or(0),
                None,
            ),
            (None, Some(downloaded_bytes), None) => (DownloadPhase::Paused {}, downloaded_bytes, None),
            (None, None, None) => (DownloadPhase::NotDownloaded {}, 0, None),
        };
        DownloadState::new(config, phase, downloaded_bytes, total_bytes)
    }

    async fn verify(
        &self,
        config: &DownloadConfig,
    ) -> Result<u64, VerifyError> {
        let actual = fs::asyn::file_length(&config.destination).await?;
        if let Some(expected) = config.expected_bytes
            && expected != actual
        {
            return Err(VerifyError::Size {
                expected,
                actual,
            });
        }
        let Some(crc) = &config.expected_crc32c else {
            return Ok(actual);
        };
        if CrcReceipt::matches(&config.destination, crc).await {
            return Ok(actual);
        }
        if !crc.verify(&config.destination).await? {
            return Err(VerifyError::Crc);
        }
        let _ = CrcReceipt::save(&config.destination, crc).await;
        Ok(actual)
    }

    async fn remove_files(
        &self,
        config: &DownloadConfig,
    ) {
        let _ = fs::asyn::remove_file(&config.resume_artifact_path).await;
        let _ = fs::asyn::remove_file(&config.destination).await;
        CrcReceipt::remove(&config.destination).await;
    }
}
