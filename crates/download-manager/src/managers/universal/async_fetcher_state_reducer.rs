use crate::{
    CheckedFileState, FileDownloadPhase, FileDownloadState, FileState, InternalDownloadState, LockFileState, Path,
    calculate_and_verify_crc, check_lock_file, crc_utils::save_crc_file, fs,
};

pub type PartFileState = FileState;

pub fn check_file_exists(path: &Path) -> FileState {
    if path.exists() && path.is_file() {
        tracing::debug!("[AF REDUCE] check_file_exists: EXISTS path={}", path.display());
        FileState::Exists
    } else {
        tracing::debug!("[AF REDUCE] check_file_exists: MISSING path={}", path.display());
        FileState::Missing
    }
}

pub fn check_crc_file_exists(path: &Path) -> FileState {
    if path.exists() && path.is_file() {
        FileState::Exists
    } else {
        FileState::Missing
    }
}

pub fn check_part_file_exists(path: &Path) -> PartFileState {
    if path.exists() && path.is_file() {
        FileState::Exists
    } else {
        FileState::Missing
    }
}

pub fn reduce_to_checked_file_state(
    downloaded_file_state: FileState,
    crc_file_state: FileState,
    destination: &Path,
    expected_bytes: Option<u64>,
    expected_crc: Option<&str>,
) -> CheckedFileState {
    tracing::info!(
        "[AF REDUCE_CHECKED] file={:?}, crc_file={:?}, expected_crc={:?}, path={}",
        downloaded_file_state,
        crc_file_state,
        expected_crc,
        destination.display()
    );
    match (downloaded_file_state, crc_file_state, expected_crc) {
        // File missing - nothing to validate
        (FileState::Missing, _, _) => CheckedFileState::Missing,

        // File exists with CRC file - use cached CRC (fast path)
        (FileState::Exists, FileState::Exists, Some(expected_crc_value)) => {
            let crc_path = format!("{}.crc", destination.display());
            if let Ok(saved_crc) = std::fs::read_to_string(&crc_path) {
                let file_size_matches_expected = expected_bytes
                    .is_some_and(|bytes| destination.metadata().is_ok_and(|metadata| metadata.len() == bytes));
                if saved_crc.trim() == expected_crc_value && file_size_matches_expected {
                    tracing::debug!("[AF REDUCE_CHECKED] ✓ Using cached CRC for {}", destination.display());
                    // Cached CRC matches - fast path!
                    CheckedFileState::Valid
                } else {
                    tracing::warn!(
                        "[AF REDUCE_CHECKED] CRC mismatch in cache, recalculating for {}",
                        destination.display()
                    );
                    // CRC mismatch in cache - recalculate to be sure
                    if let Ok(is_valid) = calculate_and_verify_crc(destination, expected_crc_value) {
                        if is_valid {
                            tracing::debug!(
                                "[AF REDUCE_CHECKED] ✓ CRC recalculation passed; updating cache for {}",
                                destination.display()
                            );
                            // Update cache with correct value
                            let _ = save_crc_file(destination, expected_crc_value);
                            CheckedFileState::Valid
                        } else {
                            tracing::warn!(
                                "[AF REDUCE_CHECKED] ✗ CRC recalculation failed for {}",
                                destination.display()
                            );
                            CheckedFileState::Invalid
                        }
                    } else {
                        tracing::warn!("[AF REDUCE_CHECKED] ✗ CRC recalculation errored for {}", destination.display());
                        CheckedFileState::Invalid
                    }
                }
            } else {
                tracing::warn!(
                    "[AF REDUCE_CHECKED] Failed to read cached CRC, recalculating for {}",
                    destination.display()
                );
                // Failed to read cached CRC - recalculate
                if let Ok(is_valid) = calculate_and_verify_crc(destination, expected_crc_value) {
                    if is_valid {
                        tracing::debug!(
                            "[AF REDUCE_CHECKED] ✓ CRC recalculation passed; caching for {}",
                            destination.display()
                        );
                        // Save the cache for next time
                        let _ = save_crc_file(destination, expected_crc_value);
                        CheckedFileState::Valid
                    } else {
                        tracing::warn!("[AF REDUCE_CHECKED] ✗ CRC recalculation failed for {}", destination.display());
                        CheckedFileState::Invalid
                    }
                } else {
                    tracing::warn!("[AF REDUCE_CHECKED] ✗ CRC recalculation errored for {}", destination.display());
                    CheckedFileState::Invalid
                }
            }
        },

        // File exists without CRC file - recalculate and cache
        (FileState::Exists, FileState::Missing, Some(expected_crc_value)) => {
            if let Ok(is_valid) = calculate_and_verify_crc(destination, expected_crc_value) {
                if is_valid {
                    tracing::debug!("[AF REDUCE_CHECKED] ✓ CRC valid; caching for {}", destination.display());
                    // Cache the CRC for next time
                    let _ = save_crc_file(destination, expected_crc_value);
                    CheckedFileState::Valid
                } else {
                    tracing::warn!("[AF REDUCE_CHECKED] ✗ CRC invalid for {}", destination.display());
                    CheckedFileState::Invalid
                }
            } else {
                tracing::warn!("[AF REDUCE_CHECKED] ✗ CRC calculation errored for {}", destination.display());
                CheckedFileState::Invalid
            }
        },

        // File exists without CRC requirement - consider valid
        (FileState::Exists, _, None) => CheckedFileState::Valid,
    }
}

pub async fn reconcile_to_internal_state(
    checked_file_state: CheckedFileState,
    part_file_state: PartFileState,
    destination: &Path,
    crc_file_path: &Path,
    part_file_path: &Path,
    _expected_bytes: Option<u64>,
    expected_crc: Option<&str>,
) -> InternalDownloadState<()> {
    tracing::info!(
        "[AF RECONCILE] checked={:?}, part={:?}, path={}, expected_bytes={:?}",
        checked_file_state,
        part_file_state,
        destination.display(),
        _expected_bytes
    );
    match checked_file_state {
        CheckedFileState::Valid => {
            // CRC file should already exist (created during reduce_to_checked_file_state)
            // but save it if somehow missing
            if let Some(crc_value) = expected_crc {
                if !crc_file_path.exists() {
                    let _ = save_crc_file(destination, crc_value);
                }
            }
            let _ = fs::remove_file(part_file_path).await;
            InternalDownloadState::Downloaded
        },
        CheckedFileState::Invalid => {
            let _ = fs::remove_file(destination).await;
            let _ = fs::remove_file(crc_file_path).await;
            match part_file_state {
                FileState::Exists => InternalDownloadState::Paused {
                    part_path: part_file_path.to_path_buf(),
                },
                FileState::Missing => InternalDownloadState::NotDownloaded,
            }
        },
        CheckedFileState::Missing => match part_file_state {
            FileState::Exists => InternalDownloadState::Paused {
                part_path: part_file_path.to_path_buf(),
            },
            FileState::Missing => InternalDownloadState::NotDownloaded,
        },
    }
}

pub fn reduce_to_file_download_state(
    checked_file_state: CheckedFileState,
    part_file_state: PartFileState,
    destination: &Path,
    part_file_path: &Path,
    expected_bytes: Option<u64>,
    manager_id: &str,
) -> FileDownloadState {
    tracing::info!(
        "[AF REDUCE_STATE] checked={:?}, part={:?}, path={}, expected_bytes={:?}",
        checked_file_state,
        part_file_state,
        destination.display(),
        expected_bytes
    );

    // Lock awareness: if the file is locked by another manager/process,
    // reflect this in the user-facing state. Lock decision belongs to reduction.
    let lock_path = std::path::PathBuf::from(format!("{}.lock", destination.display()));
    let lock_state = check_lock_file(&lock_path, manager_id, std::process::id());
    tracing::debug!(
        "[AF REDUCE_STATE] lock_path={}, lock_state={:?}, manager_id={}",
        lock_path.display(),
        lock_state,
        manager_id
    );
    if let LockFileState::OwnedByOtherApp(info) = lock_state {
        if !matches!(checked_file_state, CheckedFileState::Valid) {
            let total = expected_bytes.unwrap_or(0);
            let state = FileDownloadState {
                phase: FileDownloadPhase::LockedByOther(info.manager_id),
                downloaded_bytes: 0,
                total_bytes: total,
            };
            tracing::info!("[AF REDUCE_STATE] → Result: {:?}", state);
            return state;
        }
    }

    let state = match checked_file_state {
        CheckedFileState::Valid => {
            let file_size = std::fs::metadata(destination).ok().map(|m| m.len()).unwrap_or(0);
            FileDownloadState {
                phase: FileDownloadPhase::Downloaded,
                downloaded_bytes: file_size,
                total_bytes: file_size,
            }
        },
        CheckedFileState::Invalid => match part_file_state {
            FileState::Exists => {
                let part_size = std::fs::metadata(part_file_path).ok().map(|m| m.len()).unwrap_or(0);
                let total = expected_bytes.unwrap_or(part_size);
                FileDownloadState {
                    phase: FileDownloadPhase::Paused,
                    downloaded_bytes: part_size,
                    total_bytes: total,
                }
            },
            FileState::Missing => {
                let total = expected_bytes.unwrap_or(0);
                FileDownloadState {
                    phase: FileDownloadPhase::NotDownloaded,
                    downloaded_bytes: 0,
                    total_bytes: total,
                }
            },
        },
        CheckedFileState::Missing => match part_file_state {
            FileState::Exists => {
                let part_size = std::fs::metadata(part_file_path).ok().map(|m| m.len()).unwrap_or(0);
                let total = expected_bytes.unwrap_or(part_size);
                FileDownloadState {
                    phase: FileDownloadPhase::Paused,
                    downloaded_bytes: part_size,
                    total_bytes: total,
                }
            },
            FileState::Missing => {
                let total = expected_bytes.unwrap_or(0);
                FileDownloadState {
                    phase: FileDownloadPhase::NotDownloaded,
                    downloaded_bytes: 0,
                    total_bytes: total,
                }
            },
        },
    };

    tracing::info!("[AF REDUCE_STATE] → Result: {:?}", state);
    state
}
