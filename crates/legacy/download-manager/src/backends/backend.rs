use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use kiban::fs;

use crate::{
    DownloadPhase, DownloadState,
    backends::{ActiveTask, BackendError, BackendEventSender, DownloadGeneration},
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

    async fn reconcile(
        &self,
        config: &DownloadConfig,
    ) -> Result<(DownloadState, Option<DestinationLock>), BackendError> {
        let destination_exists = fs::asyn::is_file(&config.destination).await;
        let untouched = !destination_exists
            && !fs::asyn::is_file(&config.resume_artifact_path).await
            && !DestinationLock::exists(&config.destination).await;
        // A download acquires its destination lock before creating a native
        // task. A process crash releases the OS lock but leaves the file behind.
        if untouched {
            return Ok((DownloadState::new(config, DownloadPhase::NotDownloaded {}, 0, None), None));
        }
        if let Some(owner) = DestinationLock::foreign_owner(&config.destination, &config.owner).await {
            return Ok((self.observe(config, Some(owner)).await, None));
        }
        // Completed files need only a size check, never native task discovery.
        let pending_task = if destination_exists && self.completed_size(config).await.is_ok() {
            false
        } else {
            self.has_pending_task(config).await?
        };
        let lock = match DestinationLock::acquire(&config.destination, &config.owner).await {
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
            self.completed_size(config).await.ok()
        } else {
            None
        };
        if foreign_owner.is_none() {
            if downloaded.is_some() {
                let _ = fs::asyn::remove_file(&config.resume_artifact_path).await;
            } else {
                let _ = fs::asyn::remove_file(&config.destination).await;
            }
        }
        let (phase, downloaded_bytes, total_bytes) = match (downloaded, resume_bytes, foreign_owner) {
            (downloaded, downloaded_bytes, Some(manager_id)) => (
                DownloadPhase::Locked {
                    manager_id,
                },
                downloaded.or(downloaded_bytes).unwrap_or(0),
                None,
            ),
            (Some(size), _, None) => (DownloadPhase::Downloaded {}, size, Some(size)),
            (None, Some(downloaded_bytes), None) => (DownloadPhase::Paused {}, downloaded_bytes, None),
            (None, None, None) => (DownloadPhase::NotDownloaded {}, 0, None),
        };
        DownloadState::new(config, phase, downloaded_bytes, total_bytes)
    }

    async fn completed_size(
        &self,
        config: &DownloadConfig,
    ) -> Result<u64, BackendError> {
        let actual = fs::asyn::file_length(&config.destination).await?;
        if let Some(expected) = config.expected_bytes
            && expected != actual
        {
            return Err(BackendError::Size {
                expected,
                actual,
            });
        }
        Ok(actual)
    }

    async fn remove_files(
        &self,
        config: &DownloadConfig,
    ) {
        let _ = fs::asyn::remove_file(&config.resume_artifact_path).await;
        let _ = fs::asyn::remove_file(&config.destination).await;
    }
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;
    use crate::locks::LockOwner;

    struct UnavailableNativeBackend;

    #[async_trait::async_trait]
    impl Backend for UnavailableNativeBackend {
        fn name(&self) -> &'static str {
            "unavailable"
        }
        fn resume_artifact_extension(&self) -> &'static str {
            "resume_data"
        }

        async fn start(
            &self,
            _config: Arc<DownloadConfig>,
            _generation: DownloadGeneration,
            _events: BackendEventSender,
        ) -> Result<Box<dyn ActiveTask>, BackendError> {
            unreachable!()
        }

        async fn read_resume_progress(
            &self,
            _path: &Path,
        ) -> u64 {
            0
        }

        async fn has_pending_task(
            &self,
            _config: &DownloadConfig,
        ) -> Result<bool, BackendError> {
            Err(std::io::Error::other("native discovery unavailable").into())
        }

        async fn attach_pending_task(
            &self,
            _config: Arc<DownloadConfig>,
            _generation: DownloadGeneration,
            _events: BackendEventSender,
        ) -> Result<Option<Box<dyn ActiveTask>>, BackendError> {
            unreachable!()
        }
    }

    #[tokio::test]
    async fn startup_uses_metadata_and_preserves_native_recovery_markers() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("large-model.bin");
        let config = DownloadConfig {
            download_id: Uuid::new_v4(),
            source_url: "https://example.invalid/model".to_string(),
            bearer_token: None,
            resume_artifact_path: destination.with_extension("bin.resume_data"),
            destination: destination.clone(),
            expected_bytes: Some(15_000_000_000),
            owner: LockOwner {
                manager_id: "test".to_string(),
                instance_id: Uuid::new_v4(),
            },
        };
        let backend = UnavailableNativeBackend;
        let (state, _) = backend.reconcile(&config).await.unwrap();
        assert_eq!(state.phase, DownloadPhase::NotDownloaded {});

        // This sparse file has no receipt; catalog initialization must never read its contents.
        tokio::fs::File::create(&destination).await.unwrap().set_len(config.expected_bytes.unwrap()).await.unwrap();
        let (state, _) =
            tokio::time::timeout(std::time::Duration::from_secs(2), backend.reconcile(&config)).await.unwrap().unwrap();
        assert_eq!(state.phase, DownloadPhase::Downloaded {});

        let foreign = LockOwner {
            manager_id: "other".to_string(),
            instance_id: Uuid::new_v4(),
        };
        let lock = DestinationLock::acquire(&destination, &foreign).await.unwrap();
        let (state, _) = backend.reconcile(&config).await.unwrap();
        assert!(matches!(state.phase, DownloadPhase::Locked { .. }));
        assert!(destination.is_file());
        drop(lock);

        tokio::fs::remove_file(&destination).await.unwrap();
        let marker = destination.with_extension("bin.lock");
        tokio::fs::write(&marker, b"stale marker from interrupted native download").await.unwrap();
        assert!(backend.reconcile(&config).await.is_err());
        assert!(marker.is_file(), "failed discovery must retain evidence of a possible background task");
    }
}
