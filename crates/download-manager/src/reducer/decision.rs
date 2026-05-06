use crate::{
    CheckedFileState, FileState, LockFileState,
    file_download_task_actor::{ProgressCounters, PublicProjection},
    reducer::{Action, ActionPlan, DiskObservation, InitialLifecycleState, LockObservation, ValidationOutcome},
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Decision {
    pub initial_lifecycle_state: InitialLifecycleState,
    pub initial_projection: PublicProjection,
    pub initial_progress: ProgressCounters,
    pub action_plan: ActionPlan,
}

pub fn decide(
    observation: &DiskObservation,
    lock_observation: &LockObservation,
    validation: &ValidationOutcome,
) -> Decision {
    let initial_projection = match &lock_observation.state {
        LockFileState::OwnedByOtherApp(lock_file_info) => {
            PublicProjection::LockedByOther(lock_file_info.manager_id.clone())
        },
        _ => PublicProjection::None,
    };

    let action_plan = if lock_observation.state.is_conflict() {
        ActionPlan::empty()
    } else {
        let decision_action_plan = decide_actions(observation, validation);
        ActionPlan::merge_in_order([validation.action_plan.clone(), decision_action_plan])
    };

    let initial_lifecycle_state = match validation.checked {
        CheckedFileState::Valid => InitialLifecycleState::Downloaded {
            file_path: observation.destination_path.clone(),
            crc_path: observation.crc_path.clone(),
        },
        CheckedFileState::Invalid | CheckedFileState::Missing if observation.resume_state == FileState::Exists => {
            InitialLifecycleState::Paused {
                part_path: observation
                    .resume_artifact_path
                    .clone()
                    .unwrap_or_else(|| observation.destination_path.with_extension("part")),
            }
        },
        CheckedFileState::Invalid | CheckedFileState::Missing => InitialLifecycleState::NotDownloaded,
    };

    let initial_progress = match &initial_lifecycle_state {
        InitialLifecycleState::Paused {
            ..
        } => ProgressCounters {
            downloaded_bytes: observation.resume_size.unwrap_or(0),
            total_bytes: observation.expected_bytes.or(observation.resume_size).unwrap_or(0),
        },
        InitialLifecycleState::Downloaded {
            ..
        } => {
            let total_bytes = observation.expected_bytes.or(observation.destination_size).unwrap_or(0);
            ProgressCounters {
                downloaded_bytes: total_bytes,
                total_bytes,
            }
        },
        InitialLifecycleState::NotDownloaded => ProgressCounters {
            downloaded_bytes: 0,
            total_bytes: observation.expected_bytes.unwrap_or(0),
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
