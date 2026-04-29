use std::path::PathBuf;

use download_manager_v2::{
    CheckedFileState, FileState, LockFileInfo, LockFileState,
    file_download_task_actor::{ProgressCounters, PublicProjection},
    reducer::{Action, ActionPlan, DiskObservation, InitialLifecycleState, LockObservation, decide, validate},
};

fn observation(
    destination_state: FileState,
    crc_state: FileState,
    resume_state: FileState,
    expected_crc: Option<&str>,
) -> DiskObservation {
    DiskObservation {
        destination_state,
        crc_state,
        resume_state,
        destination_size: Some(100),
        resume_size: Some(40),
        expected_crc: expected_crc.map(str::to_string),
        expected_bytes: Some(120),
        destination_path: PathBuf::from("model.bin"),
        crc_path: Some(PathBuf::from("model.bin.crc")),
        resume_artifact_path: Some(PathBuf::from("model.bin.part")),
    }
}

fn lock_observation(lock_file_state: LockFileState) -> LockObservation {
    LockObservation {
        state: lock_file_state,
    }
}

#[test]
fn test_reducer_valid_file_deletes_resume_artifact_and_projects_downloaded_inputs() {
    let observation = observation(FileState::Exists, FileState::Exists, FileState::Exists, None);
    let validation = validate(&observation);
    let decision = decide(&observation, &lock_observation(LockFileState::Missing), &validation);

    assert_eq!(validation.checked, CheckedFileState::Valid);
    assert_eq!(
        decision.initial_lifecycle_state,
        InitialLifecycleState::Downloaded {
            file_path: PathBuf::from("model.bin"),
            crc_path: Some(PathBuf::from("model.bin.crc")),
        }
    );
    assert_eq!(decision.initial_projection, PublicProjection::None);
    assert_eq!(
        decision.initial_progress,
        ProgressCounters {
            downloaded_bytes: 120,
            total_bytes: 120,
        }
    );
    assert_eq!(
        decision.action_plan,
        ActionPlan::from_ordered_actions([Action::DeleteResumeArtifact {
            path: PathBuf::from("model.bin.part"),
        }])
    );
}

#[test]
fn test_reducer_missing_file_with_resume_returns_paused_inputs() {
    let observation = observation(FileState::Missing, FileState::Missing, FileState::Exists, None);
    let validation = validate(&observation);
    let decision = decide(&observation, &lock_observation(LockFileState::Missing), &validation);

    assert_eq!(validation.checked, CheckedFileState::Missing);
    assert_eq!(
        decision.initial_lifecycle_state,
        InitialLifecycleState::Paused {
            part_path: PathBuf::from("model.bin.part"),
        }
    );
    assert_eq!(
        decision.initial_progress,
        ProgressCounters {
            downloaded_bytes: 40,
            total_bytes: 120,
        }
    );
}

#[test]
fn test_reducer_invalid_file_deletes_bad_file_and_crc_but_keeps_resume() {
    let observation = observation(FileState::Exists, FileState::Missing, FileState::Exists, Some("expected"));
    let validation = validate(&observation);
    let decision = decide(&observation, &lock_observation(LockFileState::Missing), &validation);

    assert_eq!(validation.checked, CheckedFileState::Invalid);
    assert_eq!(
        decision.initial_lifecycle_state,
        InitialLifecycleState::Paused {
            part_path: PathBuf::from("model.bin.part"),
        }
    );
    assert_eq!(
        decision.action_plan,
        ActionPlan::from_ordered_actions([Action::DeleteFile {
            path: PathBuf::from("model.bin"),
        }])
    );
}

#[test]
fn test_reducer_live_external_lock_sets_public_projection_only() {
    let observation = observation(FileState::Missing, FileState::Missing, FileState::Missing, None);
    let validation = validate(&observation);
    let owner = LockFileInfo::new("other-manager".to_string(), 123);
    let decision = decide(&observation, &lock_observation(LockFileState::OwnedByOtherApp(owner)), &validation);

    assert_eq!(decision.initial_lifecycle_state, InitialLifecycleState::NotDownloaded);
    assert_eq!(decision.initial_projection, PublicProjection::LockedByOther("other-manager".to_string()));
}
