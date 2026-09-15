#[cfg(feature = "bindings-uniffi")]
uniffi::setup_scaffolding!();

mod backends;
mod bearer_token;
mod cached_download_task;
mod checksum;
mod checksum_receipt;
mod download_error;
mod download_id;
mod download_manager;
mod download_manager_type;
mod download_phase;
mod download_state;
mod download_task;
mod download_task_kind;
mod download_task_request;
mod file_download;
mod group_download_task;
mod locks;

#[cfg(target_vendor = "apple")]
pub use backends::AppleBackendError;
pub use backends::BackendError;
pub use bearer_token::BearerToken;
pub use checksum::Checksum;
pub use download_error::DownloadError;
pub use download_id::DownloadId;
pub use download_manager::DownloadManager;
pub use download_manager_type::DownloadManagerType;
pub use download_phase::DownloadPhase;
pub use download_state::DownloadState;
pub use download_task::DownloadTask;
pub use download_task_kind::DownloadTaskKind;
pub use download_task_request::DownloadTaskRequest;
pub use file_download::{FileDownloadError, FileDownloadTask};
pub use group_download_task::GroupDownloadTask;
pub use locks::{DestinationLock, LockError, LockOwner};
