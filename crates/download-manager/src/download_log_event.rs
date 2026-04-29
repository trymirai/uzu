use serde::{Deserialize, Serialize};

use crate::{DownloadId, FileDownloadEvent, file_download_task_actor::BackendEvent, traits::ActiveDownloadGeneration};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DownloadLogEvent {
    ManagerCreated {
        manager_id: String,
    },
    StartupReconciled {
        download_id: DownloadId,
    },
    TaskSpawned {
        download_id: DownloadId,
    },
    BackendProgress {
        generation: ActiveDownloadGeneration,
        downloaded_bytes: u64,
        total_bytes: Option<u64>,
    },
    BackendTerminal {
        event: BackendEvent,
    },
    PublicEventEmitted {
        download_id: DownloadId,
        event: FileDownloadEvent,
    },
    FsmTransition {
        from: &'static str,
        to: &'static str,
    },
}

pub fn record_download_log_event(event: DownloadLogEvent) {
    tracing::debug!(download_log_event = ?event, "download manager event");
}
