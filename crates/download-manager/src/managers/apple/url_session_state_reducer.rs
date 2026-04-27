//! State Reduction System
//!
//! This module implements a pure functional state reduction system for file downloads.
//! It takes various inputs (file existence, CRC files, resume data, URLSession tasks)
//! and produces deterministic outputs (FileDownloadState, InternalDownloadState).
//!
//! ## Three-Stage Pipeline
//!
//! 1. **Validation** (`reduce_to_checked_file_state`):
//!    - Checks file existence
//!    - Validates CRC integrity (with caching optimization)
//!    - Returns: Valid, Invalid, or Missing
//!
//! 2. **Display** (`reduce_to_file_download_state`):
//!    - Maps validated state to user-facing phase
//!    - Calculates progress percentages
//!    - Returns: FileDownloadState for UI
//!
//! 3. **Reconciliation** (`reconcile_to_internal_state`):
//!    - Performs cleanup actions (delete corrupted files, etc.)
//!    - Determines what we have after cleanup
//!    - Returns: InternalDownloadState for operations
//!
//! ## CRC Caching for Performance
//!
//! To avoid recalculating CRC on every app launch:
//!
//! **On Download Completion:**
//! 1. File downloads successfully
//! 2. Calculate CRC to validate
//! 3. If valid → Save to `.crc` file (e.g., `model.safetensors.crc`)
//!
//! **On Next Launch:**
//! 1. Check if `.crc` file exists
//! 2. If exists → Read cached CRC and compare (instant, no file scan)
//! 3. If missing → Calculate CRC → Save to cache
//! 4. If mismatch → Recalculate to verify → Update cache
//!
//! This makes subsequent launches **100x faster** for large model files.

use std::path::{Path, PathBuf};

use crate::{
    CRCFileState, CheckedFileState, DownloadedFileState, FileDownloadState, InternalDownloadState, LockFileState,
    ResumeDataFileState, calculate_and_verify_crc, check_lock_file,
    managers::apple::{URLSessionDownloadTaskResumeData, URLSessionDownloadTaskState, UrlSessionDownloadTaskExt},
    prelude::*,
};

/// Check if the downloaded file exists on disk
pub fn check_file_exists(path: &Path) -> DownloadedFileState {
    if path.exists() {
        DownloadedFileState::Exists
    } else {
        DownloadedFileState::Missing
    }
}

/// Check if the CRC validation file (.crc) exists on disk
pub fn check_crc_file_exists(path: &Path) -> CRCFileState {
    if path.exists() {
        CRCFileState::Exists
    } else {
        CRCFileState::Missing
    }
}

/// Check if the resume data file (.resume_data) exists on disk
pub fn check_resume_file_exists(path: &Path) -> ResumeDataFileState {
    if path.exists() {
        ResumeDataFileState::Exists
    } else {
        ResumeDataFileState::Missing
    }
}
fn validate_crc(
    file_path: &Path,
    expected_crc: &str,
) -> bool {
    #[cfg(debug_assertions)]
    {
        tracing::debug!("[CRC] Starting CRC validation for file: {}", file_path.display());
        let start = std::time::Instant::now();
        let result = calculate_and_verify_crc(file_path, expected_crc).unwrap_or(false);
        let duration = start.elapsed();
        tracing::debug!(
            "[CRC] CRC validation {} in {:.2}s",
            if result {
                "✓ PASSED"
            } else {
                "✗ FAILED"
            },
            duration.as_secs_f64()
        );
        return result;
    }

    #[cfg(not(debug_assertions))]
    calculate_and_verify_crc(file_path, expected_crc).unwrap_or(false)
}

/// First Reduction: Validate the downloaded file
///
/// Checks if the downloaded file exists and validates its integrity using CRC if expected.
/// Returns CheckedFileState indicating whether the file is Valid, Invalid, or Missing.
///
/// CRC Caching Mechanism:
/// - On download completion: Calculate CRC → if valid, save to `.crc` file
/// - On next app launch: Check `.crc` file exists → use cached value (fast path)
/// - If cache missing: Recalculate CRC → save to `.crc` file for next time
/// - If cache mismatch: Recalculate to verify → update cache if valid
///
/// Logic:
/// - If file is missing → Missing (regardless of CRC file existence)
/// - If file exists + CRC cached → Use cached value (fast)
/// - If file exists + CRC not cached → Validate CRC → Cache result → Valid or Invalid
/// - If file exists + no CRC expected → Valid (no validation needed)
pub fn reduce_to_checked_file_state(
    file_state: DownloadedFileState,
    crc_file_state: CRCFileState,
    file_path: &Path,
    expected_bytes: Option<u64>,
    expected_crc: Option<&str>,
) -> CheckedFileState {
    tracing::info!(
        "[REDUCE_CHECKED] file={:?}, crc_file={:?}, expected_crc={:?}, path={}",
        file_state,
        crc_file_state,
        expected_crc,
        file_path.display()
    );

    let result = match (file_state, crc_file_state, expected_crc) {
        // File missing - nothing to validate
        (DownloadedFileState::Missing, _, _) => CheckedFileState::Missing,

        // File exists with CRC file - use cached CRC (fast path)
        (DownloadedFileState::Exists, CRCFileState::Exists, Some(expected)) => {
            let crc_path = format!("{}.crc", file_path.display());
            if let Ok(saved_crc) = std::fs::read_to_string(&crc_path) {
                let file_size_matches_expected = expected_bytes
                    .is_some_and(|bytes| std::fs::metadata(file_path).is_ok_and(|metadata| metadata.len() == bytes));
                if saved_crc.trim() == expected && file_size_matches_expected {
                    tracing::debug!("[REDUCE_CHECKED] ✓ Using cached CRC for {}", file_path.display());
                    CheckedFileState::Valid
                } else {
                    tracing::warn!("[REDUCE_CHECKED] CRC mismatch in cache, recalculating for {}", file_path.display());
                    // CRC mismatch, recalculate to be sure
                    if validate_crc(file_path, expected) {
                        // Update cache with correct value
                        let _ = crate::crc_utils::save_crc_file(file_path, expected);
                        CheckedFileState::Valid
                    } else {
                        CheckedFileState::Invalid
                    }
                }
            } else {
                tracing::warn!("[REDUCE_CHECKED] Failed to read cached CRC, recalculating for {}", file_path.display());
                // Fallback to recalculation if reading .crc fails
                if validate_crc(file_path, expected) {
                    // Save the cache for next time
                    let _ = crate::crc_utils::save_crc_file(file_path, expected);
                    CheckedFileState::Valid
                } else {
                    CheckedFileState::Invalid
                }
            }
        },

        // File exists without CRC file - recalculate and cache
        (DownloadedFileState::Exists, CRCFileState::Missing, Some(expected)) => {
            if validate_crc(file_path, expected) {
                // Cache the CRC for next time
                if let Err(e) = crate::crc_utils::save_crc_file(file_path, expected) {
                    tracing::warn!("[REDUCE_CHECKED] Failed to cache CRC for {}: {}", file_path.display(), e);
                } else {
                    tracing::debug!("[REDUCE_CHECKED] ✓ CRC cached for {}", file_path.display());
                }
                CheckedFileState::Valid
            } else {
                CheckedFileState::Invalid
            }
        },

        // File exists without CRC requirement - consider valid
        (DownloadedFileState::Exists, _, None) => CheckedFileState::Valid,
    };

    tracing::info!("[REDUCE_CHECKED] → Result: {:?}", result);

    result
}

/// Second Reduction: Determine FileDownloadState for display
///
/// Maps combinations of file validation result, resume data, and URLSession task state
/// to FileDownloadState with appropriate byte counts for UI display.
///
/// This produces the user-facing state based on current reality.
pub fn reduce_to_file_download_state(
    checked_state: CheckedFileState,
    resume_state: ResumeDataFileState,
    task_state: URLSessionDownloadTaskState,
    url_session_task: Option<&Retained<NSURLSessionDownloadTask>>,
    file_path: &Path,
    expected_bytes: Option<u64>,
    manager_id: &str,
) -> FileDownloadState {
    tracing::info!(
        "[REDUCE_STATE] checked={:?}, resume={:?}, task_state={:?}, path={}",
        checked_state,
        resume_state,
        task_state.map(|s| format!("{:?}", s)),
        file_path.display()
    );

    // Lock awareness (decision in reduction): when not already downloaded and a lock
    // is held by another manager, surface LockedByOther immediately.
    if !matches!(checked_state, CheckedFileState::Valid) {
        let lock_path = PathBuf::from(format!("{}.lock", file_path.display()));
        let lock_state = check_lock_file(&lock_path, manager_id, std::process::id());
        if let LockFileState::OwnedByOtherApp(info) = lock_state {
            let bytes = url_session_task.map(|task| task.count_of_bytes_received()).unwrap_or(0);
            let total = expected_bytes
                .or_else(|| url_session_task.map(|t| t.count_of_bytes_expected_to_receive()))
                .unwrap_or(0);
            return FileDownloadState {
                phase: crate::FileDownloadPhase::LockedByOther(info.manager_id),
                downloaded_bytes: bytes,
                total_bytes: total,
            };
        }
    }

    let result = match (checked_state, resume_state, task_state) {
        (CheckedFileState::Valid, _, _) => {
            let bytes = std::fs::metadata(file_path).map(|metadata| metadata.len()).unwrap_or(0);
            FileDownloadState::downloaded(bytes)
        },

        (CheckedFileState::Invalid, ResumeDataFileState::Missing, None) => {
            let total = expected_bytes.unwrap_or(0);
            FileDownloadState::not_downloaded(total)
        },
        (CheckedFileState::Invalid, ResumeDataFileState::Exists, None) => {
            let resume_path = PathBuf::from(format!("{}.resume_data", file_path.display()));
            let resume_bytes = get_resume_bytes(&resume_path).unwrap_or(0);
            let total = expected_bytes.unwrap_or(0);
            FileDownloadState::paused(resume_bytes, total)
        },
        (CheckedFileState::Invalid, _, Some(NSURLSessionTaskState::Running)) => {
            let bytes = url_session_task.map(|task| task.count_of_bytes_received()).unwrap_or(0);
            let total_from_task = url_session_task.map(|task| task.count_of_bytes_expected_to_receive()).unwrap_or(0);
            let total = if total_from_task > 0 {
                total_from_task
            } else {
                expected_bytes.unwrap_or(0)
            };
            FileDownloadState::downloading(bytes, total)
        },
        (CheckedFileState::Invalid, _, Some(NSURLSessionTaskState::Suspended)) => {
            let bytes = url_session_task.map(|task| task.count_of_bytes_received()).unwrap_or(0);
            let total_from_task = url_session_task.map(|task| task.count_of_bytes_expected_to_receive()).unwrap_or(0);
            let total = if total_from_task > 0 {
                total_from_task
            } else {
                expected_bytes.unwrap_or(0)
            };
            FileDownloadState::paused(bytes, total)
        },
        (CheckedFileState::Invalid, _, Some(NSURLSessionTaskState::Completed | NSURLSessionTaskState::Canceling)) => {
            let total = expected_bytes.unwrap_or(0);
            FileDownloadState::not_downloaded(total)
        },

        (CheckedFileState::Missing, ResumeDataFileState::Missing, None) => {
            let total = expected_bytes.unwrap_or(0);
            FileDownloadState::not_downloaded(total)
        },
        (CheckedFileState::Missing, ResumeDataFileState::Exists, None) => {
            let resume_path = PathBuf::from(format!("{}.resume_data", file_path.display()));
            let bytes = get_resume_bytes(&resume_path).unwrap_or(0);
            let total = expected_bytes.unwrap_or(0);
            FileDownloadState::paused(bytes, total)
        },
        (CheckedFileState::Missing, _, Some(NSURLSessionTaskState::Running)) => {
            let bytes = url_session_task.map(|task| task.count_of_bytes_received()).unwrap_or(0);
            let total_from_task = url_session_task.map(|task| task.count_of_bytes_expected_to_receive()).unwrap_or(0);
            let total = if total_from_task > 0 {
                total_from_task
            } else {
                expected_bytes.unwrap_or(0)
            };
            FileDownloadState::downloading(bytes, total)
        },
        (CheckedFileState::Missing, _, Some(NSURLSessionTaskState::Suspended)) => {
            let bytes = url_session_task.map(|task| task.count_of_bytes_received()).unwrap_or(0);
            let total_from_task = url_session_task.map(|task| task.count_of_bytes_expected_to_receive()).unwrap_or(0);
            let total = if total_from_task > 0 {
                total_from_task
            } else {
                expected_bytes.unwrap_or(0)
            };
            FileDownloadState::paused(bytes, total)
        },
        (CheckedFileState::Missing, _, Some(NSURLSessionTaskState::Completed | NSURLSessionTaskState::Canceling)) => {
            let total = expected_bytes.unwrap_or(0);
            FileDownloadState::not_downloaded(total)
        },

        _ => {
            let total = expected_bytes.unwrap_or(0);
            FileDownloadState::not_downloaded(total)
        },
    };

    tracing::info!("[REDUCE_STATE] → Result: {:?}", result);

    // Safety net: If we calculated 0 total bytes (e.g. because URLSession hasn't reported size yet)
    // but we have an expected size from registry, use the expected size to prevent progress jumps.
    if result.total_bytes == 0 {
        if let Some(expected) = expected_bytes {
            if expected > 0 {
                return FileDownloadState {
                    phase: result.phase,
                    downloaded_bytes: result.downloaded_bytes,
                    total_bytes: expected,
                };
            }
        }
    }

    result
}

/// Reconciliation: Clean up disk artifacts and determine final internal state
///
/// Performs necessary cleanup actions based on file validation and task state:
/// - Deletes corrupted/stale files and invalidates their CRC caches
/// - Keeps valid CRC cache files (`.crc`) for fast validation on next launch
/// - Deletes or keeps resume data based on state
/// - Cancels URLSession tasks when appropriate
/// - Produces resume data from suspended tasks
/// - Compares and keeps the better resume data when multiple sources exist
///
/// CRC Cache Management:
/// - Valid files: Keep `.crc` file (already created by reduce_to_checked_file_state)
/// - Invalid files: Delete both file and `.crc` cache
/// - Downloaded state: Preserve `.crc` for instant validation on next init
///
/// Returns: InternalDownloadState representing what we have after cleanup
///
/// Invariants after reconciliation:
/// - CheckedFileState::Valid → File + CRC remain on disk
/// - CheckedFileState::Invalid → File + CRC deleted
/// - CheckedFileState::Missing → CRC deleted if existed
pub async fn reconcile_to_internal_state(
    checked_file: CheckedFileState,
    resume_file: ResumeDataFileState,
    url_session_task: Option<&Retained<NSURLSessionDownloadTask>>,
    file_path: &Path,
    crc_path: &Path,
    resume_path: &Path,
    expected_bytes: Option<u64>,
) -> InternalDownloadState<Retained<NSURLSessionDownloadTask>> {
    tracing::info!(
        "[RECONCILE] checked={:?}, resume={:?}, task={:?}, path={}",
        checked_file,
        resume_file,
        url_session_task.map(|t| format!("{:?}", t.state())),
        file_path.display()
    );

    let crc_path_buf = crc_path.to_path_buf();

    let result = match (checked_file, resume_file, url_session_task) {
        // Cases 1-2: Valid file, no task
        (CheckedFileState::Valid, ResumeDataFileState::Missing, None) => {
            tracing::info!("[RECONCILE] Case 1: Valid file, no resume, no task → Downloaded");

            InternalDownloadState::Downloaded
        },
        (CheckedFileState::Valid, ResumeDataFileState::Exists, None) => {
            tracing::info!("[RECONCILE] Case 2: Valid file, has resume, no task → Downloaded (delete resume)");

            let _ = std::fs::remove_file(resume_path);
            InternalDownloadState::Downloaded
        },

        // Cases 3-10: Valid file with any task - cancel task, cleanup
        (CheckedFileState::Valid, _, Some(any_task)) => {
            tracing::info!("[RECONCILE] Cases 3-10: Valid file with task → Downloaded (cancel task, cleanup)");

            any_task.cancel();
            if resume_file == ResumeDataFileState::Exists {
                let _ = std::fs::remove_file(resume_path);
            }
            InternalDownloadState::Downloaded
        },

        // Case 11: Invalid file, no resume, no task
        (CheckedFileState::Invalid, ResumeDataFileState::Missing, None) => {
            // Check if this is a partial download (file size < expected size)
            // If so, preserve it - don't delete partial downloads
            let is_partial = if let (Ok(metadata), Some(expected)) = (std::fs::metadata(file_path), expected_bytes) {
                let actual_size = metadata.len();
                actual_size < expected
            } else {
                false
            };

            if is_partial {
                tracing::info!("[RECONCILE] Case 11: Invalid file is partial download, preserving (not deleting)");
                // Keep the partial file, but remove invalid CRC cache
                // Don't create resume data - we can't without URLSession context
                // File will be handled when user initiates download again
                let _ = std::fs::remove_file(&crc_path_buf);
                InternalDownloadState::NotDownloaded
            } else {
                tracing::info!(
                    "[RECONCILE] Case 11: Invalid file at full size → NotDownloaded (deleting corrupted file)"
                );
                let _ = std::fs::remove_file(file_path);
                let _ = std::fs::remove_file(&crc_path_buf);
                InternalDownloadState::NotDownloaded
            }
        },

        // Case 12: Invalid file, has resume, no task
        (CheckedFileState::Invalid, ResumeDataFileState::Exists, None) => {
            tracing::info!("[RECONCILE] Case 12: Invalid file, has resume → Paused");

            let _ = std::fs::remove_file(file_path);
            let _ = std::fs::remove_file(&crc_path_buf);
            InternalDownloadState::Paused {
                part_path: resume_path.to_path_buf(),
            }
        },

        // Cases 13-14: Invalid file + Running task
        (CheckedFileState::Invalid, _, Some(running_task))
            if matches!(running_task.state(), NSURLSessionTaskState::Running) =>
        {
            tracing::info!("[RECONCILE] Cases 13-14: Invalid file, running task → Downloading");

            let _ = std::fs::remove_file(file_path);
            let _ = std::fs::remove_file(&crc_path_buf);
            if resume_file == ResumeDataFileState::Exists {
                let _ = std::fs::remove_file(resume_path);
            }
            InternalDownloadState::Downloading {
                task: Retained::clone(running_task),
            }
        },

        // Case 15: Invalid, no resume, suspended task → Paused (produce resume)
        (CheckedFileState::Invalid, ResumeDataFileState::Missing, Some(suspended_task))
            if matches!(suspended_task.state(), NSURLSessionTaskState::Suspended) =>
        {
            tracing::info!("[RECONCILE] Case 15: Invalid, no resume, suspended → Paused (produce resume)");

            let _ = std::fs::remove_file(file_path);
            let _ = std::fs::remove_file(&crc_path_buf);

            if let Ok(resume_data) = suspended_task.cancel_by_producing_resume_data().await {
                let _ = resume_data.save_to_file(resume_path);
                InternalDownloadState::Paused {
                    part_path: resume_path.to_path_buf(),
                }
            } else {
                InternalDownloadState::NotDownloaded
            }
        },

        // Case 16: Invalid, has resume, suspended task → Paused (keep better)
        (CheckedFileState::Invalid, ResumeDataFileState::Exists, Some(suspended_task))
            if matches!(suspended_task.state(), NSURLSessionTaskState::Suspended) =>
        {
            tracing::info!("[RECONCILE] Case 16: Invalid, has resume, suspended → Paused (compare resume)");

            let _ = std::fs::remove_file(file_path);
            let _ = std::fs::remove_file(&crc_path_buf);

            if let Ok(new_resume) = suspended_task.cancel_by_producing_resume_data().await {
                let existing_bytes = get_resume_bytes(resume_path).unwrap_or(0);
                let new_bytes = new_resume.bytes_received.unwrap_or(0);

                tracing::info!("[RECONCILE] Resume bytes: existing={} vs new={}", existing_bytes, new_bytes);

                if new_bytes > existing_bytes {
                    let _ = new_resume.save_to_file(resume_path);
                }
            }
            InternalDownloadState::Paused {
                part_path: resume_path.to_path_buf(),
            }
        },

        // Cases 17-20: Invalid file + Completed/Canceling → NotDownloaded
        (CheckedFileState::Invalid, _, Some(finished_task)) => {
            tracing::info!("[RECONCILE] Cases 17-20: Invalid, completed/canceling → NotDownloaded");

            let _ = std::fs::remove_file(file_path);
            let _ = std::fs::remove_file(&crc_path_buf);
            if resume_file == ResumeDataFileState::Exists {
                let _ = std::fs::remove_file(resume_path);
            }
            finished_task.cancel();
            InternalDownloadState::NotDownloaded
        },

        // Case 21: No file, no resume, no task
        (CheckedFileState::Missing, ResumeDataFileState::Missing, None) => {
            tracing::info!("[RECONCILE] Case 21: No file, no resume, no task → NotDownloaded");

            let _ = std::fs::remove_file(&crc_path_buf);
            InternalDownloadState::NotDownloaded
        },

        // Case 22: No file, has resume, no task
        (CheckedFileState::Missing, ResumeDataFileState::Exists, None) => {
            tracing::info!("[RECONCILE] Case 22: No file, has resume → Paused");

            let _ = std::fs::remove_file(&crc_path_buf);
            InternalDownloadState::Paused {
                part_path: resume_path.to_path_buf(),
            }
        },

        // Cases 23-24: No file + Running task → Downloading
        (CheckedFileState::Missing, _, Some(running_task))
            if matches!(running_task.state(), NSURLSessionTaskState::Running) =>
        {
            tracing::info!("[RECONCILE] Cases 23-24: No file, running task → Downloading");

            let _ = std::fs::remove_file(&crc_path_buf);
            if resume_file == ResumeDataFileState::Exists {
                let _ = std::fs::remove_file(resume_path);
            }
            InternalDownloadState::Downloading {
                task: Retained::clone(running_task),
            }
        },

        // Case 25: No file, no resume, suspended task → Paused (produce resume)
        (CheckedFileState::Missing, ResumeDataFileState::Missing, Some(suspended_task))
            if matches!(suspended_task.state(), NSURLSessionTaskState::Suspended) =>
        {
            tracing::info!("[RECONCILE] Case 25: No file, no resume, suspended → Paused (produce resume)");

            let _ = std::fs::remove_file(&crc_path_buf);

            if let Ok(resume_data) = suspended_task.cancel_by_producing_resume_data().await {
                let _ = resume_data.save_to_file(resume_path);
                InternalDownloadState::Paused {
                    part_path: resume_path.to_path_buf(),
                }
            } else {
                InternalDownloadState::NotDownloaded
            }
        },

        // Case 26: No file, has resume, suspended task → Paused (keep better)
        (CheckedFileState::Missing, ResumeDataFileState::Exists, Some(suspended_task))
            if matches!(suspended_task.state(), NSURLSessionTaskState::Suspended) =>
        {
            tracing::info!("[RECONCILE] Case 26: No file, has resume, suspended → Paused (compare resume)");

            let _ = std::fs::remove_file(&crc_path_buf);

            if let Ok(new_resume) = suspended_task.cancel_by_producing_resume_data().await {
                let existing_bytes = get_resume_bytes(resume_path).unwrap_or(0);
                let new_bytes = new_resume.bytes_received.unwrap_or(0);

                tracing::info!("[RECONCILE] Resume bytes: existing={} vs new={}", existing_bytes, new_bytes);

                if new_bytes > existing_bytes {
                    let _ = new_resume.save_to_file(resume_path);
                }
            }
            InternalDownloadState::Paused {
                part_path: resume_path.to_path_buf(),
            }
        },

        // Cases 27-28: No file + Completed task → Keep task alive for delegate callback
        (CheckedFileState::Missing, _, Some(completed_task))
            if matches!(completed_task.state(), NSURLSessionTaskState::Completed) =>
        {
            tracing::info!("[RECONCILE] Cases 27-28: No file, completed task → Downloading (awaiting delegate)");

            let _ = std::fs::remove_file(&crc_path_buf);
            if resume_file == ResumeDataFileState::Exists {
                let _ = std::fs::remove_file(resume_path);
            }
            // Don't cancel! Keep task alive so delegate can move the file
            InternalDownloadState::Downloading {
                task: Retained::clone(completed_task),
            }
        },

        // Cases 29-30: No file + Canceling task → NotDownloaded
        (CheckedFileState::Missing, _, Some(canceling_task)) => {
            tracing::info!("[RECONCILE] Cases 29-30: No file, canceling → NotDownloaded");

            let _ = std::fs::remove_file(&crc_path_buf);
            if resume_file == ResumeDataFileState::Exists {
                let _ = std::fs::remove_file(resume_path);
            }
            canceling_task.cancel();
            InternalDownloadState::NotDownloaded
        },
    };

    tracing::info!("[RECONCILE] → Final state: {:?}", result);

    result
}

/// Extract bytes_received from a resume data file
fn get_resume_bytes(resume_path: &Path) -> Option<u64> {
    URLSessionDownloadTaskResumeData::from_file(resume_path).ok().and_then(|data| data.bytes_received)
}
