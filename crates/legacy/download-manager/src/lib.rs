mod checked_file_state;
mod crc_utils;
mod download_error;
mod download_info;
mod download_log_event;
mod download_state;
mod file_check;
mod file_download_event;
mod file_download_group;
mod file_download_group_state;
mod file_download_manager;
mod file_download_phase;
mod file_download_request;
mod file_download_snapshot;
mod file_download_state;
mod file_download_task;
mod file_state;
mod http_download_request;
mod lock_file_info;
mod lock_file_state;
mod lock_manager;
mod recovery_metadata;
mod relative_file_path;

pub(crate) mod backends;
pub(crate) mod file_download_task_actor;
pub(crate) mod reducer;
pub(crate) mod traits;

pub use checked_file_state::CheckedFileState;
pub use crc_utils::integrity_cache_matches;
pub use download_error::{DownloadCleanupFailure, DownloadError};
pub use download_info::DownloadInfo;
#[allow(deprecated)]
pub use download_state::DownloadState;
pub use file_check::FileCheck;
pub use file_download_event::FileDownloadEvent;
pub use file_download_group::{DownloadAttempt, FileDownloadGroup};
pub use file_download_group_state::{
    FileDownloadFailure, FileDownloadGroupError, FileDownloadGroupOperation, FileDownloadGroupPhase,
    FileDownloadGroupState,
};
pub use file_download_manager::{
    DownloadEvent, DownloadEventSender, FileDownloadManager, FileDownloadManagerType, SharedDownloadEventSender,
};
pub use file_download_phase::FileDownloadPhase;
pub use file_download_request::{FileDownloadGroupSpec, FileDownloadGroupSpecError, FileDownloadRequest};
pub use file_download_snapshot::FileDownloadSnapshot;
pub use file_download_state::FileDownloadState;
pub use file_download_task::FileDownloadTask;
pub use file_state::FileState;
pub use http_download_request::{HttpDownloadRequest, RequestHeaders};
pub use lock_file_info::LockFileInfo;
pub use lock_file_state::LockFileState;
pub use lock_manager::{acquire_lock, check_lock_file, release_lock_if_owned, try_acquire_lock};
pub use relative_file_path::{RelativeFilePath, RelativeFilePathError};

pub type DownloadId = uuid::Uuid;

pub fn compute_download_id(destination_path: &std::path::Path) -> DownloadId {
    let normalized = normalized_destination_identity(destination_path);

    #[cfg(target_os = "macos")]
    {
        use std::os::unix::ffi::OsStrExt;

        use unicode_normalization::UnicodeNormalization;

        if let Some(path) = normalized.to_str() {
            let portable = path.nfd().flat_map(char::to_lowercase).collect::<String>();
            return uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, portable.as_bytes());
        }
        uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, normalized.as_os_str().as_bytes())
    }
    #[cfg(windows)]
    {
        use unicode_normalization::UnicodeNormalization;

        let portable =
            normalized.to_string_lossy().replace('\\', "/").nfd().flat_map(char::to_lowercase).collect::<String>();
        uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, portable.as_bytes())
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        use std::os::unix::ffi::OsStrExt;
        uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, normalized.as_os_str().as_bytes())
    }
    #[cfg(not(any(unix, windows)))]
    uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, normalized.to_string_lossy().as_bytes())
}

fn normalized_destination_identity(path: &std::path::Path) -> std::path::PathBuf {
    use std::path::{Component, PathBuf};

    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else if let Ok(current_dir) = std::env::current_dir() {
        current_dir.join(path)
    } else {
        path.to_path_buf()
    };
    let mut lexical = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => lexical.push(component),
            Component::CurDir => {},
            Component::ParentDir => {
                lexical.pop();
            },
        }
    }

    #[cfg(not(target_family = "wasm"))]
    {
        let mut ancestor = lexical.clone();
        let mut missing_suffix = Vec::new();
        loop {
            if let Ok(mut canonical) = std::fs::canonicalize(&ancestor) {
                for component in missing_suffix.iter().rev() {
                    canonical.push(component);
                }
                return canonical;
            }
            let Some(name) = ancestor.file_name().map(ToOwned::to_owned) else {
                break;
            };
            missing_suffix.push(name);
            if !ancestor.pop() {
                break;
            }
        }
    }

    lexical
}

#[cfg(test)]
#[path = "../tests/unit/download_id_test.rs"]
mod download_id_tests;

#[cfg(test)]
extern crate self as download_manager;
#[cfg(test)]
#[path = "../tests/unit/common/mod.rs"]
mod common;
#[cfg(test)]
#[expect(deprecated, reason = "compatibility tests exercise the deprecated task API")]
#[path = "../tests/unit/mod.rs"]
mod unit_tests;
