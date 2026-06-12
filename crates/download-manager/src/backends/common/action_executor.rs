use std::path::Path;

use crate::{
    DownloadError,
    crc_utils::save_crc_file,
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
            | Action::DeleteCrcCache {
                path,
            }
            | Action::DeleteResumeArtifact {
                path,
            } => {
                remove_file_if_present(path).await?;
            },
            Action::SaveCrcCache {
                destination,
                crc,
            } => {
                save_crc_file(destination, crc)?;
            },
        }
    }

    Ok(())
}

async fn remove_file_if_present(path: &Path) -> Result<(), DownloadError> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(DownloadError::from(error)),
    }
}
