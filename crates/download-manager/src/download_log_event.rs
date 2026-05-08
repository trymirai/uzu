use serde::{Deserialize, Serialize};

use crate::{DownloadId, FileDownloadEvent, file_download_task_actor::BackendEvent, traits::ActiveDownloadGeneration};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DownloadLogEvent {
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
    PublicEventEmitted {
        download_id: DownloadId,
        event: FileDownloadEvent,
    },
}

pub(crate) fn log(event: DownloadLogEvent) {
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
