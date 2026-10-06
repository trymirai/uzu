use tauri::{AppHandle, Manager};
use uzu::{session::chat::ChatSession, types::session::chat::ChatConfig};

use super::{
    ChatState, ResidentSession,
    payloads::SamplingDefaults,
    session_events::{LoadingStatus, emit_session_loading, emit_session_state},
};
use crate::{
    engine::engine,
    error::{AppError, AppResult},
    models::{ReasoningSupport, find_chat_model, reasoning_support},
};

pub(super) fn arm_idle_timer(
    app: &AppHandle,
    state: &ChatState,
    repo_id: &str,
) {
    let (enabled, idle_ms) = {
        let cfg = state.idle_config.lock().expect("idle_config mutex poisoned");
        (cfg.enabled, cfg.idle_ms)
    };
    state.cancel_idle_timer();
    if !enabled {
        return;
    }
    let app = app.clone();
    let repo_id = repo_id.to_string();
    let handle = tauri::async_runtime::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(idle_ms)).await;
        let state = app.state::<ChatState>();
        state.evict(&app, &repo_id).await;
    });
    *state.idle_timer.lock().expect("idle_timer mutex poisoned") = Some(handle);
}

pub(super) async fn ensure_session(
    app: &AppHandle,
    state: &ChatState,
    repo_id: &str,
) -> AppResult<(ChatSession, ReasoningSupport)> {
    let model = find_chat_model(repo_id).await?;
    let engine = engine().await?;
    let download_state = engine.download_state(&model).await;
    let downloaded =
        download_state.as_ref().is_some_and(|s| matches!(s.phase, uzu::storage::DownloadPhase::Downloaded {}));
    if model.is_downloadable() && !downloaded {
        return Err(AppError::msg("Model is not downloaded"));
    }
    let mut guard = state.session.lock().await;
    if let Some(resident) = guard.as_ref()
        && resident.identifier == model.identifier
    {
        return Ok((resident.session.clone(), resident.support.clone()));
    }
    let support = reasoning_support(&model);
    *guard = None;
    state.forget_history();
    // Loading events only on the real load path: emitting them on session reuse
    // would flash the spinner and the resident-model chip.
    emit_session_loading(app, LoadingStatus::Start, repo_id, None);
    let config = ChatConfig::create();
    let session = match engine.chat(model.clone(), config).await {
        Ok(session) => session,
        Err(error) => {
            emit_session_loading(app, LoadingStatus::Error, repo_id, Some(error.to_string()));
            return Err(error.into());
        },
    };
    // uzu stores these as f32; widening leaves 0.6 as 0.6000000238.
    let round = |v: Option<f64>| v.map(|v| (v * 10_000.0).round() / 10_000.0);
    let sampling_defaults = session.sampling_defaults().await.map(|p| SamplingDefaults {
        temperature: round(p.temperature),
        top_k: p.top_k,
        top_p: round(p.top_p),
        min_p: round(p.min_p),
        repetition_penalty: round(p.repetition_penalty),
        suffix_repetition_length: p.suffix_repetition_length,
    });
    *guard = Some(ResidentSession {
        repo_id: repo_id.to_string(),
        identifier: model.identifier.clone(),
        session: session.clone(),
        support: support.clone(),
        sampling_defaults,
    });
    emit_session_state(app, true, repo_id, false);
    Ok((session, support))
}
