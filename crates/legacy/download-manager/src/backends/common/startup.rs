use std::{path::Path, sync::Arc};

use kiban::fs;

use crate::{
    DownloadError, LockFileState,
    backends::common::{Backend, DownloadConfig, action_executor::apply_actions},
    check_lock_file,
    crc_utils::crc_path_for_file,
    file_state::FileState,
    lock_manager::{DestinationLockLease, lock_path_for_destination},
    reducer::{Decision, DiskObservation, LockObservation, decide, validate},
};

#[derive(Clone, Debug)]
pub struct Startup {
    pub config: Arc<DownloadConfig>,
    pub decision: Decision,
    pub lock_state: LockFileState,
}

impl Startup {
    pub async fn observe<B: Backend>(config: Arc<DownloadConfig>) -> Result<Self, DownloadError> {
        let destination_path = config.destination.as_path();
        let resume_artifact_path = destination_path.with_extension(B::RESUME_ARTIFACT_EXTENSION);
        let crc_path = crc_path_for_file(destination_path);
        let resume_state = file_state(&resume_artifact_path).await;
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
            expected_crc: config.file_check.expected_crc(),
            expected_bytes: config.expected_bytes,
            destination_path: destination_path.to_path_buf(),
            crc_path: Some(crc_path),
            resume_artifact_path: Some(resume_artifact_path),
        };
        let lock_state = check_lock_file(
            &lock_path_for_destination(destination_path),
            &config.manager_id,
            config.manager_instance_id,
            kiban::process::id(),
        )
        .await;
        let lock_observation = LockObservation {
            state: lock_state.clone(),
        };
        let validation = validate(&observation).await;
        let decision = decide(&observation, &lock_observation, &validation);

        Ok(Self {
            config,
            decision,
            lock_state,
        })
    }

    pub async fn apply_actions(
        &self,
        destination_lease: &DestinationLockLease,
    ) -> Result<(), DownloadError> {
        apply_actions(&self.decision.action_plan, destination_lease).await
    }
}

async fn file_state(path: &Path) -> FileState {
    if fs::asyn::try_exists(path).await.is_ok() && fs::asyn::is_file(path).await {
        FileState::Exists
    } else {
        FileState::Missing
    }
}
