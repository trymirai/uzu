use std::path::PathBuf;

use download_manager::{
    CheckedFileState, FileState,
    reducer::{DiskObservation, validate},
};

fn observation(
    destination_size: u64,
    expected_bytes: u64,
) -> DiskObservation {
    DiskObservation {
        destination_state: FileState::Exists,
        crc_state: FileState::Missing,
        resume_state: FileState::Missing,
        destination_size: Some(destination_size),
        resume_size: None,
        expected_crc: None,
        expected_bytes: Some(expected_bytes),
        destination_path: PathBuf::from("model.bin"),
        crc_path: None,
        resume_artifact_path: None,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn validate_rejects_existing_file_when_size_does_not_match_expected_bytes() {
    let validation = validate(&observation(100, 120)).await;
    assert_eq!(
        validation.checked,
        CheckedFileState::Invalid,
        "existing file with wrong size and no CRC must be Invalid",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn validate_accepts_existing_file_when_size_matches_expected_bytes() {
    let validation = validate(&observation(120, 120)).await;
    assert_eq!(
        validation.checked,
        CheckedFileState::Valid,
        "existing file with matching size and no CRC must be Valid",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn validate_accepts_existing_file_when_expected_bytes_unknown() {
    let mut obs = observation(100, 120);
    obs.expected_bytes = None;
    let validation = validate(&obs).await;
    assert_eq!(
        validation.checked,
        CheckedFileState::Valid,
        "without expected_bytes the validator can't reject; behaves like before",
    );
}
