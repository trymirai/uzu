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

pub fn validate(observation: &DiskObservation) -> ValidationOutcome {
    let (checked, mut actions) = match (observation.destination_state, observation.expected_crc.as_deref()) {
        (FileState::Missing, _) => (CheckedFileState::Missing, Vec::new()),
        (FileState::Exists, None) => (CheckedFileState::Valid, Vec::new()),
        (FileState::Exists, Some(expected_crc)) if cached_crc_matches(observation, expected_crc) => {
            (CheckedFileState::Valid, Vec::new())
        },
        (FileState::Exists, Some(expected_crc)) => {
            match calculate_and_verify_crc(&observation.destination_path, expected_crc) {
                Ok(true) => (
                    CheckedFileState::Valid,
                    vec![Action::SaveCrcCache {
                        destination: observation.destination_path.clone(),
                        crc: expected_crc.to_string(),
                    }],
                ),
                Ok(false) | Err(_) => (CheckedFileState::Invalid, Vec::new()),
            }
        },
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

fn cached_crc_matches(
    observation: &DiskObservation,
    expected_crc: &str,
) -> bool {
    if observation.crc_state == FileState::Missing {
        return false;
    }

    observation
        .crc_path
        .as_ref()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .is_some_and(|saved_crc| saved_crc.trim() == expected_crc)
}
