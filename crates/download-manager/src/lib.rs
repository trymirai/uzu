#[cfg(feature = "bindings-uniffi")]
uniffi::setup_scaffolding!();

mod checked_file_state;
mod crc_utils;
mod download_error;
mod download_info;
mod download_log_event;
mod download_state;
mod file_check;
mod file_download_event;
mod file_download_manager;
mod file_download_phase;
mod file_download_state;
mod file_download_task;
mod file_state;
mod lock_file_info;
mod lock_file_state;
mod lock_manager;

pub(crate) mod backends;
pub(crate) mod file_download_task_actor;
pub(crate) mod reducer;
pub(crate) mod traits;

pub use checked_file_state::CheckedFileState;
pub use download_error::DownloadError;
pub use download_info::DownloadInfo;
pub use download_state::DownloadState;
pub use file_check::FileCheck;
pub use file_download_event::FileDownloadEvent;
pub use file_download_manager::{
    DownloadEvent, DownloadEventSender, FileDownloadManager, FileDownloadManagerType, SharedDownloadEventSender,
};
pub use file_download_phase::FileDownloadPhase;
pub use file_download_state::FileDownloadState;
pub use file_download_task::FileDownloadTask;
pub use file_state::FileState;
pub use lock_file_info::LockFileInfo;
pub use lock_file_state::LockFileState;
pub use lock_manager::{acquire_lock, check_lock_file, release_lock_if_owned, try_acquire_lock};

pub type DownloadId = uuid::Uuid;
pub fn compute_download_id(destination_path: &std::path::Path) -> DownloadId {
    uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, destination_path.display().to_string().as_bytes())
}
