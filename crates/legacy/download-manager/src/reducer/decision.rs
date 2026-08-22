use crate::{
    CheckedFileState, FileState, LockFileState,
    file_download_task_actor::{ProgressCounters, PublicProjection},
    reducer::{Action, ActionPlan, DiskObservation, InitialLifecycleState, ValidationOutcome},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub initial_lifecycle_state: InitialLifecycleState,
    pub initial_projection: PublicProjection,
    pub initial_progress: ProgressCounters,
    pub action_plan: ActionPlan,
}

pub fn decide(
    observation: &DiskObservation,
    lock_state: &LockFileState,
    validation: &ValidationOutcome,
) -> Decision {
    let action_plan = if lock_state.is_conflict() {
        ActionPlan::empty()
    } else {
        let decision_action_plan = decide_actions(observation, validation);
        ActionPlan::merge_in_order([validation.action_plan.clone(), decision_action_plan])
    };

    let initial_lifecycle_state =
        match (&validation.checked, &observation.resume_state, observation.resume_artifact_path.as_ref()) {
            (CheckedFileState::Valid, _, _) => InitialLifecycleState::Downloaded,
            (CheckedFileState::Invalid | CheckedFileState::Missing, FileState::Exists, Some(part_path)) => {
                InitialLifecycleState::Paused {
                    part_path: part_path.clone(),
                }
            },
            (CheckedFileState::Invalid | CheckedFileState::Missing, _, _) => InitialLifecycleState::NotDownloaded,
        };

    // Don't surface `LockedByOther` when the file is already on disk; ownership-sensitive
    // Lock ownership is projected through each task snapshot.
    let initial_projection = match (lock_state, &initial_lifecycle_state) {
        (LockFileState::OwnedByOtherApp(lock_file_info), state)
            if !matches!(state, InitialLifecycleState::Downloaded) =>
        {
            PublicProjection::LockedByOther(lock_file_info.manager_id.clone())
        },
        _ => PublicProjection::None,
    };

    let initial_progress = match &initial_lifecycle_state {
        InitialLifecycleState::Paused {
            ..
        } => ProgressCounters {
            downloaded_bytes: observation.resume_size.unwrap_or(0),
            total_bytes: observation.expected_bytes,
        },
        InitialLifecycleState::Downloaded => {
            let total_bytes = observation.expected_bytes.or(observation.destination_size);
            ProgressCounters {
                downloaded_bytes: total_bytes.unwrap_or(0),
                total_bytes,
            }
        },
        InitialLifecycleState::NotDownloaded => ProgressCounters {
            downloaded_bytes: 0,
            total_bytes: observation.expected_bytes,
        },
    };

    Decision {
        initial_lifecycle_state,
        initial_projection,
        initial_progress,
        action_plan,
    }
}

fn decide_actions(
    observation: &DiskObservation,
    validation: &ValidationOutcome,
) -> ActionPlan {
    let mut actions = Vec::new();

    if validation.checked == CheckedFileState::Valid
        && observation.resume_state == FileState::Exists
        && let Some(path) = observation.resume_artifact_path.clone()
    {
        actions.push(Action::DeleteResumeArtifact {
            path,
        });
    }

    if validation.checked == CheckedFileState::Invalid {
        actions.push(Action::DeleteFile {
            path: observation.destination_path.clone(),
        });

        if observation.crc_state == FileState::Exists
            && let Some(path) = observation.crc_path.clone()
        {
            actions.push(Action::DeleteCrcCache {
                path,
            });
        }
    }

    ActionPlan::from_ordered_actions(actions)
}
