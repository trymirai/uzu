use download_manager::{FileDownloadPhase, FileDownloadState};
use uzu::storage::types::{DownloadPhase, reduce_file_download_states};

#[test]
fn errored_file_surfaces_even_when_other_files_are_still_downloading() {
    let file_states = vec![
        FileDownloadState {
            phase: FileDownloadPhase::Error("crc mismatch".to_string()),
            downloaded_bytes: 0,
            total_bytes: 1_000,
        },
        FileDownloadState {
            phase: FileDownloadPhase::Downloading,
            downloaded_bytes: 500,
            total_bytes: 1_000,
        },
    ];

    let reduced = reduce_file_download_states(&file_states);

    assert!(
        matches!(reduced.phase, DownloadPhase::Error { .. }),
        "any errored file must surface as model-level Error; got {:?}",
        reduced.phase,
    );
}

#[test]
fn errored_file_surfaces_when_other_files_are_downloaded() {
    let file_states = vec![
        FileDownloadState {
            phase: FileDownloadPhase::Downloaded,
            downloaded_bytes: 1_000,
            total_bytes: 1_000,
        },
        FileDownloadState {
            phase: FileDownloadPhase::Error("network timeout".to_string()),
            downloaded_bytes: 0,
            total_bytes: 2_000,
        },
    ];

    let reduced = reduce_file_download_states(&file_states);

    assert!(
        matches!(reduced.phase, DownloadPhase::Error { .. }),
        "errored file must surface as model-level Error even alongside completed files; got {:?}",
        reduced.phase,
    );
}

#[test]
fn all_downloaded_remains_downloaded() {
    let file_states = vec![
        FileDownloadState {
            phase: FileDownloadPhase::Downloaded,
            downloaded_bytes: 1_000,
            total_bytes: 1_000,
        },
        FileDownloadState {
            phase: FileDownloadPhase::Downloaded,
            downloaded_bytes: 2_000,
            total_bytes: 2_000,
        },
    ];

    let reduced = reduce_file_download_states(&file_states);

    assert!(matches!(reduced.phase, DownloadPhase::Downloaded {}));
    assert_eq!(reduced.total_bytes, 3_000);
}
