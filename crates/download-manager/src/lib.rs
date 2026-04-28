mod checked_file_state;
mod crc_utils;
mod download_error;
mod download_info;
mod download_manager_state;
mod download_state;
mod file_check;
mod file_download_event;
mod file_download_manager;
mod file_download_phase;
mod file_download_state;
mod file_download_task;
mod file_state;
mod internal_download_state;
mod lock_file_info;
mod lock_file_state;
mod lock_manager;
pub mod managers;
mod prelude;
mod utils;

pub use checked_file_state::CheckedFileState;
pub use download_error::DownloadError;
pub use download_info::DownloadInfo;
pub use download_state::DownloadState;
pub use file_check::FileCheck;
pub use file_download_event::FileDownloadEvent;
pub use file_download_manager::{FileDownloadManager, FileDownloadManagerType, create_download_manager};
pub use file_download_phase::FileDownloadPhase;
pub use file_download_state::FileDownloadState;
pub use file_download_task::FileDownloadTask;
pub use file_state::{CRCFileState, DownloadedFileState, FileState, ResumeDataFileState};
pub use internal_download_state::{InternalDownloadState, StateTransitionAction};
pub use lock_file_info::LockFileInfo;
pub use lock_file_state::LockFileState;
pub use lock_manager::{acquire_lock, check_lock_file, release_lock, try_acquire_lock};
pub use utils::compute_download_id;

pub type TaskID = usize;
pub type DownloadId = uuid::Uuid;

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

// Re-export for internal use (used by lock modules)
#[allow(unused_imports)]
pub(crate) use chrono;
use crc_utils::calculate_and_verify_crc;
// Import NSBundle on all Apple platforms (needed for manager ID generation)
#[cfg(target_vendor = "apple")]
use objc2_foundation::NSBundle;
#[allow(unused_imports)]
pub(crate) use serde;
#[allow(unused_imports)]
pub(crate) use serde_json;
use tokio::{
    fs,
    runtime::Handle as TokioHandle,
    sync::{
        Mutex as TokioMutex,
        broadcast::{Sender as TokioBroadcastSender, channel as tokio_broadcast_channel},
    },
    task::JoinHandle as TokioJoinHandle,
};
use uuid::Uuid;
