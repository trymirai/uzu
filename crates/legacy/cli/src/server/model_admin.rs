//! Admin API for the model slot: query status, load, unload.
//!
//! - `GET  /v1/models/status` — slot state (idle/loading/ready/failed).
//! - `POST /v1/models/load`   — resolve + load a model into the slot (202,
//!   background task; loads are serialized by the load gate).
//! - `POST /v1/models/unload` — clear the slot and drain the resident model.
//!
//! Swap semantics (see `model_slot.rs` for memory lifetimes): the slot goes to
//! `Loading` immediately (chat requests get 409 from that point), then the
//! task waits for the old model's text session mutex so any in-flight
//! generation finishes (bounded by [`MODEL_SWAP_DRAIN_TIMEOUT`]), drops the
//! old model, and builds the new one. Queued-but-not-started requests that
//! snapshotted `Ready` before the swap may still run against the old session
//! after the swap — single-user semantics; their Arc keeps the old model
//! alive until they finish.

use std::sync::Arc;
use std::time::Duration;

use rocket::{State, get, post, response::status, serde::json::Json};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uzu::engine::Engine;

use crate::{
    server::{
        ChatState, SwapSlot, error::{OaiError, OaiErrorResponse},
        loader::{LoadParams, load_model}, model_slot::SlotSnapshot, request_info::RequestInfo,
    },
};

/// How long a swap waits for an in-flight generation on the old model to
/// finish before giving up (the slot then reports `failed` with this error).
const MODEL_SWAP_DRAIN_TIMEOUT: Duration = Duration::from_secs(120);

// ---------------------------------------------------------------- status ---

#[derive(Serialize)]
pub struct LoadedModelInfo {
    pub model: String,
    /// RFC 3339 UTC timestamp of when the model became ready.
    pub loaded_at: String,
    pub uptime_seconds: f64,
    pub vlm: bool,
    pub audio: bool,
    pub max_context_tokens: u32,
}

#[derive(Serialize)]
pub struct LoadingModelInfo {
    pub model: String,
    pub elapsed_seconds: f64,
}

#[derive(Serialize)]
pub struct LastErrorInfo {
    pub model: String,
    pub error: String,
    /// RFC 3339 UTC timestamp of the failure.
    pub at: String,
}

#[derive(Serialize)]
pub struct ModelStatusResponse {
    /// `"idle" | "loading" | "ready" | "failed"`.
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loaded: Option<LoadedModelInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loading: Option<LoadingModelInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<LastErrorInfo>,
    pub version: String,
}

#[get("/models/status")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_model_status(
    chat: &State<ChatState>,
    request_info: &RequestInfo,
) -> Json<ModelStatusResponse> {
    let snapshot = chat.slot.snapshot().await;
    Json(match snapshot {
        SlotSnapshot::Ready(loaded) => ModelStatusResponse {
            status: "ready".to_string(),
            model: Some(loaded.model_name.clone()),
            loaded: Some(LoadedModelInfo {
                model: loaded.model_name.clone(),
                loaded_at: loaded.loaded_at.to_rfc3339(),
                uptime_seconds: loaded.loaded_at_instant.elapsed().as_secs_f64(),
                vlm: loaded.vlm_pipeline.is_some(),
                audio: loaded.audio.is_some(),
                max_context_tokens: loaded.max_context_tokens,
            }),
            loading: None,
            last_error: None,
            version: Engine::version(),
        },
        SlotSnapshot::Loading {
            model,
            started_at,
        } => ModelStatusResponse {
            status: "loading".to_string(),
            model: Some(model.clone()),
            loaded: None,
            loading: Some(LoadingModelInfo {
                model,
                elapsed_seconds: started_at.elapsed().as_secs_f64(),
            }),
            last_error: None,
            version: Engine::version(),
        },
        SlotSnapshot::Failed {
            model,
            error,
            at,
        } => ModelStatusResponse {
            status: "failed".to_string(),
            model: Some(model.clone()),
            loaded: None,
            loading: None,
            last_error: Some(LastErrorInfo {
                model,
                error,
                at: at.to_rfc3339(),
            }),
            version: Engine::version(),
        },
        SlotSnapshot::Empty => ModelStatusResponse {
            status: "idle".to_string(),
            model: None,
            loaded: None,
            loading: None,
            last_error: None,
            version: Engine::version(),
        },
    })
}

// ------------------------------------------------------------------- load ---

#[derive(Deserialize)]
pub struct ModelLoadRequest {
    /// Models-dir directory name, absolute local directory path, or catalog id.
    pub model: String,
}

#[derive(Serialize)]
pub struct ModelLoadResponse {
    /// `"ready"` (was already resident) or `"loading"` (accepted, in progress).
    pub status: String,
    pub model: String,
}

#[post("/models/load", format = "json", data = "<body>")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_model_load(
    chat: &State<ChatState>,
    body: Json<ModelLoadRequest>,
    request_info: &RequestInfo,
) -> Result<status::Custom<Json<ModelLoadResponse>>, status::Custom<Json<OaiErrorResponse>>> {
    use rocket::http::Status;

    fn load_response(status: Status, response: ModelLoadResponse) -> status::Custom<Json<ModelLoadResponse>> {
        status::Custom(status, Json(response))
    }

    let model = body.model.trim().to_string();
    if model.is_empty() {
        return Err(status::Custom(
            Status::BadRequest,
            Json(OaiErrorResponse {
                error: OaiError {
                    message: "'model' must be a non-empty model directory name, local path, or catalog id".to_string(),
                    kind: "invalid_request_error".to_string(),
                    param: Some("model".to_string()),
                    code: Some("invalid_request".to_string()),
                },
            }),
        ));
    }

    // Already resident? 200 without touching the slot.
    if let SlotSnapshot::Ready(loaded) = chat.slot.snapshot().await
        && loaded.model_name == model {
            return Ok(load_response(
                Status::Ok,
                ModelLoadResponse {
                    status: "ready".to_string(),
                    model,
                },
            ));
        }

    // Hand the work to the background task; the response only acknowledges.
    // Concurrent/queued loads are serialized by the load gate inside the task.
    let engine = Arc::clone(&chat.engine);
    let slot = Arc::clone(&chat.slot);
    let gate = Arc::clone(&chat.load_gate);
    let params = LoadParams {
        seed: chat.seed,
        max_context_tokens: chat.max_context_tokens,
    };
    tokio::spawn(load_task(engine, slot, gate, params, model.clone()));

    Ok(load_response(
        Status::Accepted,
        ModelLoadResponse {
            status: "loading".to_string(),
            model,
        },
    ))
}

/// Background load/swap: serialized by `gate`, drains the previous resident
/// model, then builds the new one. The slot transitions Loading → Ready/Failed.
async fn load_task(
    engine: Arc<Engine>,
    slot: Arc<SwapSlot>,
    gate: Arc<Mutex<()>>,
    params: LoadParams,
    target: String,
) {
    // One load at a time; queued loads proceed after the current one finishes.
    let _gate = gate.lock().await;

    // A queued load may be redundant by the time it gets the gate (an earlier
    // load for the same model succeeded); don't tear down a matching model.
    if let SlotSnapshot::Ready(loaded) = slot.snapshot().await
        && loaded.model_name == target {
            return;
        }

    let previous = slot.begin_loading(target.clone()).await;

    // Swap: drain the previous model before allocating the new one. Holding
    // the session mutex also keeps queued old-model requests from starting a
    // generation while the new model loads. run_stream/run_blocking release
    // the lock when their client disconnects, so a dead requester can't wedge
    // the swap; the timeout bounds a genuinely stuck generation.
    if let SlotSnapshot::Ready(old) = previous {
        tracing::info!("🔁 Swapping model: {} → {}", old.model_name, target);
        let drain = tokio::time::timeout(MODEL_SWAP_DRAIN_TIMEOUT, old.session.lock()).await;
        match drain {
            Ok(_session_guard) => {
                // `_session_guard` held (and the old model pinned by `old`)
                // across the load below so the old weights stay out of the
                // GPU's way while the new ones allocate; both drop after.
            },
            Err(_) => {
                slot.set_failed(
                    target,
                    format!(
                        "previous model busy: in-flight generation did not drain within {:?}",
                        MODEL_SWAP_DRAIN_TIMEOUT
                    ),
                )
                .await;
                return;
            },
        }
    }

    match load_model(&engine, &target, &params).await {
        Ok(loaded) => slot.set_ready(loaded).await,
        Err(error) => slot.set_failed(target, format!("{error:#}")).await,
    }
}

// ----------------------------------------------------------------- unload ---

#[derive(Serialize)]
pub struct ModelUnloadResponse {
    /// `"idle"` after the slot is cleared, or `"loading"` if a load is running.
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[post("/models/unload")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_model_unload(
    chat: &State<ChatState>,
    request_info: &RequestInfo,
) -> Json<ModelUnloadResponse> {
    if let SlotSnapshot::Loading { model, .. } = chat.slot.snapshot().await {
        return Json(ModelUnloadResponse {
            status: "loading".to_string(),
            detail: Some(format!("load of '{model}' in progress; unload rejected")),
        });
    }

    let previous = chat.slot.unload().await;
    match previous {
        Some(old) => {
            tracing::info!("🗑️  Unloading model: {}", old.model_name);
            // Bounded drain: usually the session is idle and this returns at
            // once. If a generation is somehow still running (e.g. the client
            // vanished mid-stream), hand the model to a background task so
            // GPU memory frees when it finishes instead of blocking the
            // response for up to the drain timeout.
            let session = Arc::clone(&old.session);
            let drained = tokio::time::timeout(MODEL_SWAP_DRAIN_TIMEOUT, session.lock()).await;
            let drained_in_time = drained.is_ok();
            drop(drained);
            if !drained_in_time {
                tokio::spawn(async move {
                    let _guard = tokio::time::timeout(MODEL_SWAP_DRAIN_TIMEOUT * 4, session.lock()).await;
                    tracing::info!("drained previously busy model after unload");
                });
            }
            Json(ModelUnloadResponse {
                status: "idle".to_string(),
                detail: None,
            })
        },
        None => Json(ModelUnloadResponse {
            status: "idle".to_string(),
            detail: None,
        }),
    }
}
