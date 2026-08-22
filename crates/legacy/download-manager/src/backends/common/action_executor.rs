use std::path::Path;

use kiban::fs;

use crate::{
    DownloadError,
    backends::common::reject_symlink_components,
    crc_utils::save_integrity_cache_at,
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
                reject_parent_symlinks(path).await?;
                remove_file_if_present(path).await?;
            },
            Action::SaveIntegrityCache {
                destination,
                receipt_path,
                file_check,
            } => {
                reject_symlink_components(destination).await?;
                save_integrity_cache_at(destination, file_check, receipt_path).await?;
            },
        }
    }

    Ok(())
}

async fn reject_parent_symlinks(path: &Path) -> Result<(), DownloadError> {
    if let Some(parent) = path.parent() {
        reject_symlink_components(parent).await?;
    }
    Ok(())
}

async fn remove_file_if_present(path: &Path) -> Result<(), DownloadError> {
    match fs::asyn::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(DownloadError::from(error)),
    }
}

#[cfg(all(test, unix))]
#[path = "../../../tests/unit/backends/common/action_executor_test.rs"]
mod tests;
