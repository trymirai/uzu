use crate::{
    CheckedFileState, FileState,
    crc_utils::calculate_and_verify_crc,
    reducer::{Action, ActionPlan, DiskObservation},
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValidationOutcome {
    pub checked: CheckedFileState,
    pub action_plan: ActionPlan,
}

pub async fn validate(observation: &DiskObservation) -> ValidationOutcome {
    let length_matches = match (observation.destination_size, observation.expected_bytes) {
        (Some(actual), Some(expected)) => actual == expected,
        _ => true,
    };

    let (checked, mut actions) = match (observation.destination_state, observation.expected_crc.as_deref()) {
        (FileState::Missing, _) => (CheckedFileState::Missing, Vec::new()),
        (FileState::Exists, _) if !length_matches => (CheckedFileState::Invalid, Vec::new()),
        (FileState::Exists, None) => (CheckedFileState::Valid, Vec::new()),
        (FileState::Exists, Some(expected_crc)) => validate_crc_with_cache(observation, expected_crc).await,
    };

    if checked == CheckedFileState::Missing
        && observation.crc_state == FileState::Exists
        && let Some(path) = observation.crc_path.clone()
    {
        actions.push(Action::DeleteCrcCache {
            path,
        });
    }

    let action_plan = ActionPlan::from_ordered_actions(actions);

    ValidationOutcome {
        checked,
        action_plan,
    }
}

async fn validate_crc_with_cache(
    observation: &DiskObservation,
    expected_crc: &str,
) -> (CheckedFileState, Vec<Action>) {
    if observation.crc_state == FileState::Exists
        && let Some(crc_path) = &observation.crc_path
        && let Ok(saved_crc) = std::fs::read_to_string(crc_path)
        && saved_crc.trim() == expected_crc
    {
        return (CheckedFileState::Valid, Vec::new());
    }

    let destination = observation.destination_path.clone();
    let expected = expected_crc.to_string();
    let crc_check = tokio::task::spawn_blocking(move || calculate_and_verify_crc(&destination, &expected))
        .await
        .ok()
        .and_then(|inner| inner.ok());

    match crc_check {
        Some(true) => (
            CheckedFileState::Valid,
            vec![Action::SaveCrcCache {
                destination: observation.destination_path.clone(),
                crc: expected_crc.to_string(),
            }],
        ),
        _ => (CheckedFileState::Invalid, Vec::new()),
    }
}
