#[cfg(feature = "bindings-uniffi")]
uniffi::setup_scaffolding!();

mod backends;
mod checked_file_state;
mod crc_utils;
mod download_error;
mod download_info;
mod download_log_event;
mod download_manager;
mod download_manager_type;
mod download_phase;
mod download_state;
mod download_task;
mod download_task_kind;
mod download_task_request;
mod file_check;
mod file_download_task_actor;
mod file_state;
mod group_download_task;
mod lock_file_info;
mod lock_file_state;
mod lock_manager;
mod reducer;
mod traits;

pub use download_error::DownloadError;
pub use download_manager::DownloadManager;
pub use download_manager_type::DownloadManagerType;
pub use download_phase::DownloadPhase;
pub use download_state::DownloadState;
pub use download_task::DownloadTask;
pub use download_task_kind::DownloadTaskKind;
pub use download_task_request::DownloadTaskRequest;
pub use file_check::FileCheck;
pub use file_download_task_actor::FileDownloadTask;
pub use group_download_task::GroupDownloadTask;
pub use lock_file_info::LockFileInfo;
pub use lock_file_state::LockFileState;
pub use lock_manager::{acquire_lock, check_lock_file, release_lock_if_owned};

pub type DownloadId = uuid::Uuid;
pub fn compute_download_id(destination_path: &std::path::Path) -> DownloadId {
    uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, destination_path.display().to_string().as_bytes())
}
