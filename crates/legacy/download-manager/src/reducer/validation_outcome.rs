use crate::{
    CheckedFileState, DownloadError, FileCheck, FileState,
    crc_utils::{VerificationError, VerificationStatus, integrity_cache_matches_at, verify_file_integrity},
    reducer::{Action, ActionPlan, DiskObservation},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationOutcome {
    pub checked: CheckedFileState,
    pub action_plan: ActionPlan,
}

pub async fn validate(observation: &DiskObservation) -> Result<ValidationOutcome, DownloadError> {
    let length_matches = match (observation.destination_size, observation.expected_bytes) {
        (Some(actual), Some(expected)) => actual == expected,
        _ => true,
    };

    let (checked, mut actions) = match (&observation.destination_state, &observation.file_check) {
        (FileState::Missing, _) => (CheckedFileState::Missing, Vec::new()),
        (FileState::Exists, _) if !length_matches => (CheckedFileState::Invalid, Vec::new()),
        (FileState::Exists, FileCheck::None) => (CheckedFileState::Valid, Vec::new()),
        (FileState::Exists, file_check) => validate_integrity_with_cache(observation, file_check).await?,
    };

    if checked == CheckedFileState::Missing
        && observation.crc_state == FileState::Exists
        && let Some(path) = observation.crc_path.clone()
    {
        actions.push(Action::DeleteCrcCache {
            path,
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
    if observation.crc_state == FileState::Exists
        && let Some(receipt_path) = observation.crc_path.as_ref()
        && integrity_cache_matches_at(&observation.destination_path, file_check, receipt_path).await
    {
        return Ok((CheckedFileState::Valid, Vec::new()));
    }

    match verify_file_integrity(&observation.destination_path, file_check).await {
        Ok(VerificationStatus::Match) => Ok((
            CheckedFileState::Valid,
            vec![Action::SaveIntegrityCache {
                destination: observation.destination_path.clone(),
                receipt_path: observation.crc_path.clone().expect("startup always supplies an integrity receipt path"),
                file_check: file_check.clone(),
            }],
        )),
        Ok(VerificationStatus::Mismatch) => Ok((CheckedFileState::Invalid, Vec::new())),
        Err(error) => Err(map_verification_error(&observation.destination_path, error)),
    }
}

fn map_verification_error(
    path: &std::path::Path,
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
