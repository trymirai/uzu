use std::{path::Path, sync::Arc};

use kiban::fs;

use crate::{
    CheckedFileState, DownloadError, DownloadId, FileCheck, FileState, HttpDownloadRequest, LockFileState,
    backends::common::{Backend, action_executor::apply_actions},
    check_lock_file,
    file_download_task_actor::{ProgressCounters, PublicProjection},
    lock_manager::DestinationLockLease,
    recovery_metadata::observe_resume_recovery,
    reducer::{Action, ActionPlan, DiskObservation, InitialLifecycleState, decide, validate},
    traits::DownloadConfig,
};

#[derive(Clone, Debug)]
pub struct Startup {
    pub config: Arc<DownloadConfig>,
    pub initial_lifecycle_state: InitialLifecycleState,
    pub initial_projection: PublicProjection,
    pub initial_progress: ProgressCounters,
    pub action_plan: ActionPlan,
    pub lock_state: LockFileState,
}

impl Startup {
    pub async fn observe<B: Backend>(
        download_id: DownloadId,
        request: HttpDownloadRequest,
        destination_path: &Path,
        artifact_root: &Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
        manager_id: &str,
        manager_instance_id: uuid::Uuid,
    ) -> Result<Self, DownloadError> {
        let config = Arc::new(DownloadConfig {
            download_id,
            request,
            destination: destination_path.to_path_buf(),
            artifact_root: artifact_root.to_path_buf(),
            file_check: file_check.clone(),
            expected_bytes,
            manager_id: manager_id.to_string(),
            manager_instance_id,
        });
        reject_symlink_components(destination_path).await?;
        ensure_owned_directory(artifact_root).await?;
        if let Some(lock_directory) = config.lock_path().parent() {
            ensure_owned_directory(lock_directory).await?;
        }
        let resume_artifact_path = config.resume_artifact_path(B::RESUME_ARTIFACT_EXTENSION);
        let crc_path = config.integrity_receipt_path();
        let recovery_observation = observe_resume_recovery(&config, &resume_artifact_path).await?;
        let resume_state = recovery_observation.resume_state;
        let resume_size = match resume_state {
            FileState::Exists => B::read_resume_progress(&resume_artifact_path).await,
            FileState::Missing => None,
        };
        let observation = DiskObservation {
            destination_state: file_state(destination_path).await,
            crc_state: file_state(&crc_path).await,
            resume_state,
            destination_size: fs::asyn::file_length(destination_path).await.ok(),
            resume_size,
            file_check: file_check.clone(),
            expected_bytes,
            destination_path: destination_path.to_path_buf(),
            crc_path: Some(crc_path),
            resume_artifact_path: Some(resume_artifact_path),
        };
        let lock_state =
            check_lock_file(&config.lock_path(), manager_id, manager_instance_id, kiban::process::id()).await;
        let validation = validate(&observation).await?;
        // Validation reads the destination and its receipt. Check both owned
        // paths again so a late ancestor swap cannot turn the resulting action
        // plan into an operation outside the downloader's roots.
        reject_symlink_components(destination_path).await?;
        reject_symlink_components(artifact_root).await?;
        let mut decision = decide(&observation, &lock_state, &validation);
        if !lock_state.is_conflict() {
            let mut recovery_actions = recovery_observation
                .cleanup_paths
                .iter()
                .cloned()
                .map(|path| Action::DeleteResumeArtifact {
                    path,
                })
                .collect::<Vec<_>>();
            if validation.checked == CheckedFileState::Valid && resume_state == FileState::Exists {
                recovery_actions.push(Action::DeleteResumeArtifact {
                    path: config.recovery_metadata_path(),
                });
                recovery_actions.push(Action::DeleteResumeArtifact {
                    path: config.recovery_metadata_staging_path(),
                });
            }
            decision.action_plan =
                ActionPlan::merge_in_order([decision.action_plan, ActionPlan::from_ordered_actions(recovery_actions)]);
        }
        Ok(Self {
            config,
            initial_lifecycle_state: decision.initial_lifecycle_state,
            initial_projection: decision.initial_projection,
            initial_progress: decision.initial_progress,
            action_plan: decision.action_plan,
            lock_state,
        })
    }

    pub(crate) async fn apply_actions(
        &self,
        destination_lease: &DestinationLockLease,
    ) -> Result<(), DownloadError> {
        apply_actions(&self.action_plan, destination_lease).await
    }
}

pub(crate) async fn ensure_owned_directory(path: &Path) -> Result<(), DownloadError> {
    reject_symlink_components(path).await?;
    fs::asyn::create_dir_all(path).await?;
    reject_symlink_components(path).await
}

pub(crate) async fn reject_symlink_components(path: &Path) -> Result<(), DownloadError> {
    let mut current = std::path::PathBuf::new();
    for component in path.components() {
        current.push(component);
        match tokio::fs::symlink_metadata(&current).await {
            Ok(metadata) if metadata.file_type().is_symlink() && !is_platform_path_alias(&current) => {
                return Err(DownloadError::Io(format!(
                    "download state path contains a symlink: {}",
                    current.display()
                )));
            },
            Ok(_) => {},
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(DownloadError::from(error)),
        }
    }
    Ok(())
}

fn is_platform_path_alias(path: &Path) -> bool {
    #[cfg(target_os = "macos")]
    {
        matches!(path.to_str(), Some("/var" | "/tmp" | "/etc"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = path;
        false
    }
}

async fn file_state(path: &Path) -> FileState {
    if fs::asyn::try_exists(path).await.is_ok() && fs::asyn::is_file(path).await {
        FileState::Exists
    } else {
        FileState::Missing
    }
}
