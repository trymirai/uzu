use serde::Serialize;
use tauri::{AppHandle, Manager};
use uzu::storage::{DownloadPhase, DownloadState};

use crate::models::PhaseKind;

#[derive(Serialize, Clone, PartialEq, Eq, Debug)]
#[serde(tag = "kind", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub(super) enum DownloadEvent {
    Progress {
        completed_bytes: i64,
        total_bytes: Option<i64>,
    },
    Done,
    Error {
        error: String,
    },
    Paused,
    Resumed,
    Deleted,
    Locked {
        locked_by: String,
    },
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(super) struct DownloadStateEvent {
    pub(super) seq: u64,
    pub(super) identifier: String,
    #[serde(flatten)]
    pub(super) event: DownloadEvent,
}

pub(super) fn events_for(
    state: &DownloadState,
    previous: Option<PhaseKind>,
) -> Vec<DownloadEvent> {
    match &state.phase {
        DownloadPhase::Downloading {} => {
            let progress = DownloadEvent::Progress {
                completed_bytes: state.downloaded_bytes,
                total_bytes: (state.total_bytes > 0).then_some(state.total_bytes),
            };
            // The engine has no distinct "resumed" phase, so entering Downloading
            // from any other phase is the signal that flips the UI back to an
            // active state before the first progress tick lands.
            if previous != Some(PhaseKind::Downloading) {
                vec![DownloadEvent::Resumed, progress]
            } else {
                vec![progress]
            }
        },
        DownloadPhase::Paused {} => vec![DownloadEvent::Paused],
        // done/error require a known prior phase: previous=None is the first
        // snapshot of an already-settled state (startup, delete re-emit), not a transition.
        DownloadPhase::Downloaded {} => {
            transition(previous, PhaseKind::Downloaded).then_some(DownloadEvent::Done).into_iter().collect()
        },
        DownloadPhase::Error {
            message,
        } => transition(previous, PhaseKind::Error)
            .then(|| DownloadEvent::Error {
                error: message.clone(),
            })
            .into_iter()
            .collect(),
        DownloadPhase::NotDownloaded {} => {
            transition(previous, PhaseKind::NotDownloaded).then_some(DownloadEvent::Deleted).into_iter().collect()
        },
        DownloadPhase::Locked {
            manager_id,
        } => transition(previous, PhaseKind::Locked)
            .then(|| DownloadEvent::Locked {
                locked_by: manager_id.clone(),
            })
            .into_iter()
            .collect(),
    }
}

fn transition(
    previous: Option<PhaseKind>,
    entered: PhaseKind,
) -> bool {
    matches!(previous, Some(p) if p != entered)
}

pub(super) fn announce(
    app: &AppHandle,
    event: &DownloadStateEvent,
) {
    match &event.event {
        DownloadEvent::Done => {
            crate::logger::info(
                "download:state",
                Some(serde_json::json!({ "identifier": event.identifier, "kind": "done" })),
            );
            notify_download(app, "Model downloaded", &event.identifier);
        },
        DownloadEvent::Error {
            error,
        } => {
            crate::logger::error(
                "download:state",
                Some(serde_json::json!({ "identifier": event.identifier, "kind": "error", "error": error })),
            );
            notify_download(app, "Download failed", error);
        },
        DownloadEvent::Locked {
            locked_by,
        } => crate::logger::info(
            "download:state",
            Some(serde_json::json!({ "identifier": event.identifier, "kind": "locked", "lockedBy": locked_by })),
        ),
        DownloadEvent::Progress {
            ..
        }
        | DownloadEvent::Paused
        | DownloadEvent::Resumed
        | DownloadEvent::Deleted => {},
    }
}

// In-view progress already shows the outcome; notify only when the window is out of sight.
fn can_notify(app: &AppHandle) -> bool {
    let Some(window) = app.get_webview_window("main") else {
        return false;
    };
    window.is_minimized().unwrap_or(false)
        || !window.is_visible().unwrap_or(true)
        || !window.is_focused().unwrap_or(true)
}

fn notify_download(
    app: &AppHandle,
    title: &str,
    body: &str,
) {
    if !can_notify(app) {
        return;
    }
    use tauri_plugin_notification::NotificationExt;
    let _ = app.notification().builder().title(title).body(body).show();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(
        phase: DownloadPhase,
        downloaded_bytes: i64,
        total_bytes: i64,
    ) -> DownloadState {
        DownloadState {
            phase,
            downloaded_bytes,
            total_bytes,
        }
    }

    fn payload(event: DownloadEvent) -> serde_json::Value {
        serde_json::to_value(DownloadStateEvent {
            seq: 7,
            identifier: "vendor/model".into(),
            event,
        })
        .expect("serializable")
    }

    #[test]
    fn events_serialize_flat_with_kind_tag() {
        assert_eq!(
            payload(DownloadEvent::Progress {
                completed_bytes: 10,
                total_bytes: None
            }),
            serde_json::json!({ "seq": 7, "identifier": "vendor/model", "kind": "progress", "completedBytes": 10, "totalBytes": null })
        );
        assert_eq!(
            payload(DownloadEvent::Done),
            serde_json::json!({ "seq": 7, "identifier": "vendor/model", "kind": "done" })
        );
        assert_eq!(
            payload(DownloadEvent::Locked {
                locked_by: "cli".into()
            }),
            serde_json::json!({ "seq": 7, "identifier": "vendor/model", "kind": "locked", "lockedBy": "cli" })
        );
    }

    #[test]
    fn entering_downloading_announces_resume_before_progress() {
        let s = state(DownloadPhase::Downloading {}, 10, 100);
        assert_eq!(
            events_for(&s, Some(PhaseKind::Paused)),
            vec![
                DownloadEvent::Resumed,
                DownloadEvent::Progress {
                    completed_bytes: 10,
                    total_bytes: Some(100)
                }
            ]
        );
        assert_eq!(
            events_for(&s, Some(PhaseKind::Downloading)),
            vec![DownloadEvent::Progress {
                completed_bytes: 10,
                total_bytes: Some(100)
            }]
        );
    }

    #[test]
    fn settled_phases_are_events_only_as_transitions() {
        let done = state(DownloadPhase::Downloaded {}, 100, 100);
        assert_eq!(events_for(&done, None), vec![]);
        assert_eq!(events_for(&done, Some(PhaseKind::Downloaded)), vec![]);
        assert_eq!(events_for(&done, Some(PhaseKind::Downloading)), vec![DownloadEvent::Done]);

        let failed = state(
            DownloadPhase::Error {
                message: "The network connection was lost.".into(),
            },
            0,
            0,
        );
        assert_eq!(events_for(&failed, None), vec![]);
        assert_eq!(
            events_for(&failed, Some(PhaseKind::Downloading)),
            vec![DownloadEvent::Error {
                error: "The network connection was lost.".into()
            }]
        );

        let locked = state(
            DownloadPhase::Locked {
                manager_id: "cli".into(),
            },
            0,
            0,
        );
        assert_eq!(
            events_for(&locked, Some(PhaseKind::Paused)),
            vec![DownloadEvent::Locked {
                locked_by: "cli".into()
            }]
        );
    }
}
