use serde::Serialize;
use tauri::{AppHandle, Emitter};

#[derive(Serialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum LoadingStatus {
    Start,
    Error,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct SessionLoadingEvent {
    status: LoadingStatus,
    repo_id: String,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct SessionStateEvent {
    active: bool,
    repo_id: String,
    is_ejecting: bool,
}

pub fn emit_session_loading(
    app: &AppHandle,
    status: LoadingStatus,
    repo_id: &str,
    error: Option<String>,
) {
    crate::logger::info(
        "session:loading",
        Some(serde_json::json!({ "status": status, "repoId": repo_id, "error": error })),
    );
    let _ = app.emit(
        "session-loading",
        SessionLoadingEvent {
            status,
            repo_id: repo_id.to_string(),
        },
    );
}

pub fn emit_session_state(
    app: &AppHandle,
    active: bool,
    repo_id: &str,
    is_ejecting: bool,
) {
    crate::logger::info(
        "session:state",
        Some(serde_json::json!({ "active": active, "repoId": repo_id, "isEjecting": is_ejecting })),
    );
    let _ = app.emit(
        "session-state",
        SessionStateEvent {
            active,
            repo_id: repo_id.to_string(),
            is_ejecting,
        },
    );
}
