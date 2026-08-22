#![allow(dead_code, unused_imports)]

use std::{path::Path, sync::Arc, time::Duration};

use download_manager::{
    FileCheck, FileDownloadPhase, FileDownloadSnapshot, FileDownloadState, FileDownloadTask, HttpDownloadRequest,
    compute_download_id, recovery_metadata::write_recovery_metadata, traits::DownloadConfig,
};
pub use mock_registry::{Behavior, MockRegistry};
use tokio::{sync::watch::Receiver as TokioWatchReceiver, time::timeout};

/// Resolves once the task reaches a terminal snapshot (downloaded, errored, or locked).
pub async fn wait_for_terminal(task: &Arc<dyn FileDownloadTask>) {
    let mut snapshots = task.snapshot_receiver();
    loop {
        let is_terminal = {
            let snapshot = snapshots.borrow_and_update();
            snapshot.failure.is_some()
                || matches!(
                    snapshot.state.phase,
                    FileDownloadPhase::Downloaded | FileDownloadPhase::Error(_) | FileDownloadPhase::LockedByOther(_)
                )
        };
        if is_terminal || snapshots.changed().await.is_err() {
            return;
        }
    }
}

pub async fn wait_for_phase(
    task: &Arc<dyn FileDownloadTask>,
    snapshots: &mut TokioWatchReceiver<FileDownloadSnapshot>,
    mut is_expected_phase: impl FnMut(&FileDownloadPhase) -> bool,
) -> FileDownloadState {
    timeout(Duration::from_secs(15), async {
        let state = task.state().await;
        if is_expected_phase(&state.phase) {
            return state;
        }

        while snapshots.changed().await.is_ok() {
            let state = snapshots.borrow_and_update().state.clone();
            if is_expected_phase(&state.phase) {
                return state;
            }
        }

        panic!("download snapshot watch closed before expected phase");
    })
    .await
    .expect("timed out waiting for file download phase")
}

pub fn error_message(state: FileDownloadState) -> String {
    let FileDownloadPhase::Error(message) = state.phase else {
        unreachable!("state must be an error")
    };
    message
}

pub async fn write_file_with_parents(
    path: &Path,
    contents: impl AsRef<[u8]>,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(path, contents).await
}

pub async fn write_recoverable_resume_artifact(
    resume_artifact: &Path,
    destination: &Path,
    source_url: &str,
    file_check: FileCheck,
    expected_bytes: Option<u64>,
    contents: impl AsRef<[u8]>,
) -> Result<(), download_manager::DownloadError> {
    write_file_with_parents(resume_artifact, contents).await?;
    let download_id = compute_download_id(destination);
    let config = DownloadConfig {
        download_id,
        request: HttpDownloadRequest::get(source_url),
        destination: destination.to_path_buf(),
        artifact_root: resume_artifact.parent().expect("resume artifact must have a parent").to_path_buf(),
        file_check,
        expected_bytes,
        manager_id: "test-manager".to_string(),
        manager_instance_id: uuid::Uuid::nil(),
    };
    write_recovery_metadata(&config).await
}
