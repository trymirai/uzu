use crate::{
    DownloadError,
    crc_utils::save_crc_file,
    reducer::{Action, ActionPlan},
};

pub async fn apply_actions(action_plan: &ActionPlan) -> Result<(), DownloadError> {
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
                let _ = tokio::fs::remove_file(path).await;
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
