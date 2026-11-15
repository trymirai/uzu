use std::{sync::Arc, time::Instant};

use chrono::{DateTime, Utc};
use rocket::{State, get, serde::json::Json};
use serde::Serialize;
use uzu::engine::Engine;

use crate::{
    common::thinking::ThinkingSupport,
    server::request_info::RequestInfo,
};

/// Managed Rocket state shared by every handler.
pub struct ServerState {
    /// Model passed at boot (`--model`); `None` when booted idle for dynamic
    /// loads. Decision branches always set it.
    pub model_name: Option<String>,
    /// JSONL access journal; counters stay live even when file writing is off.
    pub access_log: Arc<crate::server::access_log::AccessLog>,
    pub started_at: DateTime<Utc>,
    pub uptime: Instant,
}

#[derive(Serialize)]
pub struct VlmInfo {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    pub active_sessions: usize,
    pub max_sessions: usize,
    pub session_ttl_seconds: u64,
}

#[derive(Serialize)]
pub struct RequestCounters {
    pub started: u64,
    pub completed: u64,
    pub failed: u64,
}

#[derive(Serialize)]
pub struct RequestLogInfo {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

#[derive(Serialize)]
pub struct ServerStateResponse {
    /// Resident model; `null` while the slot is idle/loading.
    pub model: Option<String>,
    pub version: String,
    /// RFC 3339 UTC timestamp of server boot.
    pub started_at: String,
    pub uptime_seconds: f64,
    pub prefix_cache: bool,
    pub max_context_tokens: Option<u32>,
    pub thinking: Option<String>,
    pub vlm: VlmInfo,
    pub requests: RequestCounters,
    pub request_log: RequestLogInfo,
}

fn thinking_label(support: &ThinkingSupport) -> String {
    match support {
        ThinkingSupport::Levels {
            supported, ..
        } => format!(
            "levels: {}",
            supported.iter().map(|effort| format!("{effort:?}")).collect::<Vec<_>>().join(", ")
        ),
        ThinkingSupport::Toggle(enabled) => format!("toggle: {enabled}"),
        ThinkingSupport::AlwaysOn => "always-on".to_string(),
        ThinkingSupport::Unsupported => "unsupported".to_string(),
    }
}

/// Server internals: identity, uptime, resident model (from the slot),
/// VLM session pool, and the access-log request counters.
#[get("/state")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_server_state(
    state: &State<ServerState>,
    chat: &State<crate::server::ChatState>,
    request_info: &RequestInfo,
) -> Json<ServerStateResponse> {
    let (started, completed, failed) = state.access_log.counters();
    let slot = chat.slot.snapshot().await;
    let (model, max_context_tokens, thinking, vlm_info) = match slot {
        crate::server::SlotSnapshot::Ready(loaded) => {
            let vlm = loaded.vlm_pipeline.as_ref();
            let info = VlmInfo {
                enabled: vlm.is_some(),
                backend: vlm.map(|pipeline| pipeline.vision_backend().to_string()),
                active_sessions: loaded.vlm_sessions.session_count().await,
                max_sessions: loaded.vlm_sessions.max_sessions(),
                session_ttl_seconds: loaded.vlm_sessions.session_ttl().as_secs(),
            };
            (
                Some(loaded.model_name.clone()),
                Some(loaded.max_context_tokens),
                Some(thinking_label(&loaded.thinking_support)),
                info,
            )
        },
        _ => (
            None,
            None,
            None,
            VlmInfo {
                enabled: false,
                backend: None,
                active_sessions: 0,
                max_sessions: 0,
                session_ttl_seconds: 0,
            },
        ),
    };
    Json(ServerStateResponse {
        model,
        version: Engine::version(),
        started_at: state.started_at.to_rfc3339(),
        uptime_seconds: state.uptime.elapsed().as_secs_f64(),
        prefix_cache: chat.prefix_cache,
        max_context_tokens,
        thinking,
        vlm: vlm_info,
        requests: RequestCounters {
            started,
            completed,
            failed,
        },
        request_log: RequestLogInfo {
            enabled: state.access_log.path().is_some(),
            path: state.access_log.path().map(|path| path.display().to_string()),
        },
    })
}
