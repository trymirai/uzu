use std::{io::ErrorKind, path::Path};

use kiban::fs;

use crate::{
    DownloadError,
    integrity::save_integrity_cache,
    lock_manager::DestinationLockLease,
    reducer::{Action, ActionPlan},
};

pub async fn apply_actions(
    action_plan: &ActionPlan,
    _destination_lease: &DestinationLockLease,
) -> Result<(), DownloadError> {
    for action in action_plan.as_slice() {
        match action {
            Action::DeleteFile {
                path,
            }
            | Action::DeleteIntegrityCache {
                path,
            }
            | Action::DeleteResumeArtifact {
                path,
            } => {
                remove_file_if_present(path).await?;
            },
            Action::SaveIntegrityCache {
                destination,
                file_check,
            } => {
                save_integrity_cache(destination, file_check).await?;
            },
        }
    }

    Ok(())
}

async fn remove_file_if_present(path: &Path) -> Result<(), DownloadError> {
    match fs::asyn::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(DownloadError::from(error)),
    }
}
