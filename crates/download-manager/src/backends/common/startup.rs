use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{
    DownloadError, DownloadId, FileCheck, FileState,
    backends::common::{Backend, action_executor::apply_actions},
    check_lock_file,
    crc_utils::crc_path_for_file,
    file_download_task_actor::{ProgressCounters, PublicProjection},
    reducer::{ActionPlan, DiskObservation, InitialLifecycleState, LockObservation, decide, validate},
    traits::DownloadConfig,
};

#[derive(Clone, Debug)]
pub struct Startup {
    pub config: Arc<DownloadConfig>,
    pub initial_lifecycle_state: InitialLifecycleState,
    pub initial_projection: PublicProjection,
    pub initial_progress: ProgressCounters,
    pub action_plan: ActionPlan,
}

impl Startup {
    #[allow(clippy::ptr_arg)]
    pub fn observe<B: Backend>(
        download_id: DownloadId,
        source_url: &String,
        destination_path: &Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
        manager_id: &str,
    ) -> Result<Self, DownloadError> {
        let resume_artifact_path = destination_path.with_extension(B::RESUME_ARTIFACT_EXTENSION);
        let expected_crc = match &file_check {
            FileCheck::CRC(crc) => Some(crc.clone()),
            FileCheck::None => None,
        };
        let crc_path = crc_path_for_file(destination_path);
        let observation = DiskObservation {
            destination_state: file_state(destination_path),
            crc_state: file_state(&crc_path),
            resume_state: file_state(&resume_artifact_path),
            destination_size: file_size(destination_path),
            resume_size: file_size(&resume_artifact_path),
            expected_crc,
            expected_bytes,
            destination_path: destination_path.to_path_buf(),
            crc_path: Some(crc_path),
            resume_artifact_path: Some(resume_artifact_path),
        };
        let lock_observation = LockObservation {
            state: check_lock_file(&lock_path_for_destination(destination_path), manager_id, std::process::id()),
        };
        let validation = validate(&observation);
        let decision = decide(&observation, &lock_observation, &validation);
        let config = Arc::new(DownloadConfig {
            download_id,
            source_url: source_url.clone(),
            destination: destination_path.to_path_buf(),
            file_check,
            expected_bytes,
            manager_id: manager_id.to_string(),
        });

        Ok(Self {
            config,
            initial_lifecycle_state: decision.initial_lifecycle_state,
            initial_projection: decision.initial_projection,
            initial_progress: decision.initial_progress,
            action_plan: decision.action_plan,
        })
    }

    pub async fn apply_actions(&self) -> Result<(), DownloadError> {
        apply_actions(&self.action_plan).await
    }
}

fn file_state(path: &Path) -> FileState {
    if path.exists() {
        FileState::Exists
    } else {
        FileState::Missing
    }
}

fn file_size(path: &Path) -> Option<u64> {
    path.metadata().ok().map(|metadata| metadata.len())
}

fn lock_path_for_destination(destination: &Path) -> PathBuf {
    PathBuf::from(format!("{}.lock", destination.display()))
}
