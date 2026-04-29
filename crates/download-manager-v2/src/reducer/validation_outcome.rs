use crate::{
    CheckedFileState, FileState,
    crc_utils::{calculate_and_verify_crc, save_crc_file},
    reducer::{Action, ActionPlan, DiskObservation},
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValidationOutcome {
    pub checked: CheckedFileState,
    pub action_plan: ActionPlan,
}

pub fn validate(observation: &DiskObservation) -> ValidationOutcome {
    let checked = match (observation.destination_state, observation.expected_crc.as_deref()) {
        (FileState::Missing, _) => CheckedFileState::Missing,
        (FileState::Exists, None) => CheckedFileState::Valid,
        (FileState::Exists, Some(expected_crc)) if cached_crc_matches(observation, expected_crc) => {
            CheckedFileState::Valid
        },
        (FileState::Exists, Some(expected_crc)) => {
            match calculate_and_verify_crc(&observation.destination_path, expected_crc) {
                Ok(true) => {
                    let _ = save_crc_file(&observation.destination_path, expected_crc);
                    CheckedFileState::Valid
                },
                Ok(false) | Err(_) => CheckedFileState::Invalid,
            }
        },
    };

    let action_plan = if checked == CheckedFileState::Missing && observation.crc_state == FileState::Exists {
        match observation.crc_path.clone() {
            Some(path) => ActionPlan::from_ordered_actions([Action::DeleteCrcCache {
                path,
            }]),
            None => ActionPlan::empty(),
        }
    } else {
        ActionPlan::empty()
    };

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
