use serde::{Deserialize, Serialize};

use crate::{DownloadId, backends::common::ActiveDownloadGeneration, file_download_task_actor::BackendEvent};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DownloadLogEvent {
    ManagerCreated {
        manager_id: String,
    },
    StartupReconciled {
        download_id: DownloadId,
        initial_lifecycle_state: &'static str,
        action_count: usize,
    },
    TaskSpawned {
        download_id: DownloadId,
    },
    StateTransition {
        download_id: DownloadId,
        from: &'static str,
        to: &'static str,
    },
    BackendProgress {
        download_id: DownloadId,
        generation: ActiveDownloadGeneration,
        downloaded_bytes: u64,
        total_bytes: Option<u64>,
    },
    BackendTerminal {
        download_id: DownloadId,
        event: BackendEvent,
    },
}

pub fn log(event: DownloadLogEvent) {
    match &event {
        DownloadLogEvent::BackendProgress {
            ..
        } => {
            tracing::trace!(download_log_event = ?event, "download manager event");
        },
        _ => {
            tracing::debug!(download_log_event = ?event, "download manager event");
        },
    }
}
