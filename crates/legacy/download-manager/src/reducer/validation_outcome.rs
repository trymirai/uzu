use std::path::Path;

use crate::{
    CheckedFileState, DownloadError, FileCheck, FileState,
    integrity::{VerificationError, integrity_cache_matches, verify_file_integrity},
    reducer::{Action, ActionPlan, DiskObservation},
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValidationOutcome {
    pub checked: CheckedFileState,
    pub action_plan: ActionPlan,
}

pub async fn validate(observation: &DiskObservation) -> Result<ValidationOutcome, DownloadError> {
    let length_matches = match (observation.destination_size, observation.expected_bytes) {
        (Some(actual), Some(expected)) => actual == expected,
        _ => true,
    };

    let (checked, mut actions) = match (observation.destination_state, &observation.file_check) {
        (FileState::Missing, _) => (CheckedFileState::Missing, Vec::new()),
        (FileState::Exists, _) if !length_matches => (CheckedFileState::Invalid, Vec::new()),
        (FileState::Exists, FileCheck::None) => (CheckedFileState::Valid, Vec::new()),
        (FileState::Exists, file_check) => validate_integrity_with_cache(observation, file_check).await?,
    };

    if checked == CheckedFileState::Missing && observation.integrity_state == FileState::Exists {
        actions.push(Action::DeleteIntegrityCache {
            path: observation.integrity_path.clone(),
        });
    }

    Ok(ValidationOutcome {
        checked,
        action_plan: ActionPlan::from_ordered_actions(actions),
    })
}

async fn validate_integrity_with_cache(
    observation: &DiskObservation,
    file_check: &FileCheck,
) -> Result<(CheckedFileState, Vec<Action>), DownloadError> {
    if observation.integrity_state == FileState::Exists
        && integrity_cache_matches(&observation.destination_path, file_check).await
    {
        return Ok((CheckedFileState::Valid, Vec::new()));
    }

    match verify_file_integrity(&observation.destination_path, file_check).await {
        Ok(true) => Ok((
            CheckedFileState::Valid,
            vec![Action::SaveIntegrityCache {
                destination: observation.destination_path.clone(),
                file_check: file_check.clone(),
            }],
        )),
        Ok(false) => Ok((CheckedFileState::Invalid, Vec::new())),
        Err(error) => Err(map_verification_error(&observation.destination_path, error)),
    }
}

fn map_verification_error(
    path: &Path,
    error: VerificationError,
) -> DownloadError {
    match error {
        VerificationError::InvalidExpectedDigest {
            algorithm,
        } => DownloadError::InvalidDigest {
            algorithm,
        },
        VerificationError::Io(error) => DownloadError::IntegrityIo {
            path: path.display().to_string(),
            message: error.to_string(),
        },
    }
}
