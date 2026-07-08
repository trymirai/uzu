use crate::{
    CheckedFileState, FileState,
    crc_utils::{calculate_and_verify_crc, crc_cache_matches},
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
    if observation.crc_state == FileState::Exists && crc_cache_matches(&observation.destination_path, expected_crc) {
        return (CheckedFileState::Valid, Vec::new());
    }

    let crc_check = calculate_and_verify_crc(&observation.destination_path, expected_crc).await.ok();

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
