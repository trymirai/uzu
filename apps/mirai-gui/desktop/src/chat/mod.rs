mod messages;
mod payloads;
mod session;
mod session_events;
mod stream;
mod title;

use std::collections::HashMap;

use payloads::{MsgIn, RunEvent, RunStreamPayload, SamplingDefaults, TitleGenPayload};
use session::arm_idle_timer;
use session_events::emit_session_state;
use stream::run_stream_inner;
use tauri::{AppHandle, Manager, ipc::Channel};
use title::{TITLE_GEN_RUN_ID, title_gen_inner};
use uzu::{
    session::chat::ChatSession,
    types::basic::{CancelToken, ReasoningEffort},
};

use crate::{error::AppResult, models::ReasoningSupport};

const MIN_AUTO_EJECT_IDLE_MS: u64 = 60_000;
const MAX_AUTO_EJECT_IDLE_MS: u64 = 24 * 60 * 60_000;
const DEFAULT_AUTO_EJECT_IDLE_MS: u64 = 2 * 60_000;

struct IdleConfig {
    enabled: bool,
    idle_ms: u64,
}

// The exact message prefix currently represented by the session's KV cache.
#[derive(Default)]
struct SessionHistory {
    repo_id: String,
    messages: Vec<MsgIn>,
    reasoning_effort: Option<ReasoningEffort>,
}

struct ResidentSession {
    repo_id: String,
    identifier: String,
    session: ChatSession,
    support: ReasoningSupport,
    // Read right after load: uzu holds the instance lock for a whole reply, so
    // asking the session later would block until generation ends.
    sampling_defaults: Option<SamplingDefaults>,
}

// A run is registered before its model loads: a cold load takes tens of
// seconds, and a Stop that arrives in that window has no stream token yet.
#[derive(Default)]
struct RunCancel {
    requested: bool,
    token: Option<CancelToken>,
}

pub struct ChatState {
    session: tokio::sync::Mutex<Option<ResidentSession>>,
    // Serializes reset()+reply across run_stream and title_gen: they share one
    // ChatSession, and a concurrent reset() clobbers the other's stream.
    run_lock: tokio::sync::Mutex<()>,
    cancels: std::sync::Mutex<HashMap<String, RunCancel>>,
    idle_config: std::sync::Mutex<IdleConfig>,
    idle_timer: std::sync::Mutex<Option<tauri::async_runtime::JoinHandle<()>>>,
    history: std::sync::Mutex<SessionHistory>,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            session: tokio::sync::Mutex::new(None),
            run_lock: tokio::sync::Mutex::new(()),
            cancels: std::sync::Mutex::new(HashMap::new()),
            idle_config: std::sync::Mutex::new(IdleConfig {
                enabled: true,
                idle_ms: DEFAULT_AUTO_EJECT_IDLE_MS,
            }),
            idle_timer: std::sync::Mutex::new(None),
            history: std::sync::Mutex::new(SessionHistory::default()),
        }
    }
}

impl ChatState {
    fn register_run(
        &self,
        run_id: &str,
    ) {
        self.cancels.lock().expect("cancels mutex poisoned").insert(run_id.to_string(), RunCancel::default());
    }

    fn finish_run(
        &self,
        run_id: &str,
    ) {
        self.cancels.lock().expect("cancels mutex poisoned").remove(run_id);
    }

    fn request_cancel(
        &self,
        run_id: &str,
    ) {
        if let Some(run) = self.cancels.lock().expect("cancels mutex poisoned").get_mut(run_id) {
            run.requested = true;
            if let Some(token) = &run.token {
                token.cancel();
            }
        }
    }

    fn cancel_requested(
        &self,
        run_id: &str,
    ) -> bool {
        self.cancels.lock().expect("cancels mutex poisoned").get(run_id).is_some_and(|run| run.requested)
    }

    fn attach_cancel_token(
        &self,
        run_id: &str,
        token: CancelToken,
    ) {
        if let Some(run) = self.cancels.lock().expect("cancels mutex poisoned").get_mut(run_id) {
            if run.requested {
                token.cancel();
            }
            run.token = Some(token);
        }
    }

    fn cancel_idle_timer(&self) {
        if let Some(handle) = self.idle_timer.lock().expect("idle_timer mutex poisoned").take() {
            handle.abort();
        }
    }

    fn plan_run(
        &self,
        repo_id: &str,
        messages: &[MsgIn],
        reasoning_effort: Option<ReasoningEffort>,
    ) -> RunPlan {
        let history = self.history.lock().expect("history mutex poisoned");
        let continues = history.repo_id == repo_id
            && history.reasoning_effort == reasoning_effort
            && messages.len() > history.messages.len()
            && !history.messages.is_empty()
            && messages[..history.messages.len()] == history.messages[..];
        if continues {
            RunPlan::Continue {
                tail: messages[history.messages.len()..].to_vec(),
            }
        } else {
            RunPlan::Replay
        }
    }

    fn remember_history(
        &self,
        repo_id: &str,
        messages: Vec<MsgIn>,
        reasoning_effort: Option<ReasoningEffort>,
    ) {
        let mut history = self.history.lock().expect("history mutex poisoned");
        history.repo_id = repo_id.to_string();
        history.messages = messages;
        history.reasoning_effort = reasoning_effort;
    }

    fn forget_history(&self) {
        *self.history.lock().expect("history mutex poisoned") = SessionHistory::default();
    }

    async fn evict(
        &self,
        app: &AppHandle,
        repo_id: &str,
    ) -> bool {
        let mut guard = self.session.lock().await;
        let had_session = guard.is_some();
        if had_session {
            emit_session_state(app, false, repo_id, true);
        }
        *guard = None;
        self.forget_history();
        if had_session {
            emit_session_state(app, false, repo_id, false);
        }
        had_session
    }
}

enum RunPlan {
    Continue {
        tail: Vec<MsgIn>,
    },
    Replay,
}

// try_lock: while a load holds the session lock there is nothing to report yet,
// and the client re-asks once the session becomes resident.
#[tauri::command]
pub async fn chat_sampling_defaults(
    state: tauri::State<'_, ChatState>,
    repo_id: String,
) -> AppResult<Option<SamplingDefaults>> {
    let Ok(guard) = state.session.try_lock() else {
        return Ok(None);
    };
    Ok(guard.as_ref().filter(|resident| resident.repo_id == repo_id).and_then(|r| r.sampling_defaults.clone()))
}

#[tauri::command]
pub async fn run_stream(
    app: AppHandle,
    state: tauri::State<'_, ChatState>,
    payload: RunStreamPayload,
    on_event: Channel<RunEvent>,
) -> AppResult<()> {
    state.cancel_idle_timer();
    state.register_run(&payload.run_id);
    let result = run_stream_inner(&app, &state, &payload, &on_event).await;
    state.finish_run(&payload.run_id);
    if let Err(error) = result {
        crate::logger::error(
            "chat:run:error",
            Some(serde_json::json!({ "runId": payload.run_id, "repoId": payload.repo_id, "error": error })),
        );
        let _ = on_event.send(RunEvent::Error {
            error: error.to_string(),
        });
    }
    if state.session.lock().await.is_some() {
        arm_idle_timer(&app, &state, &payload.repo_id);
    }
    Ok(())
}

#[tauri::command]
pub async fn title_gen(
    app: AppHandle,
    state: tauri::State<'_, ChatState>,
    payload: TitleGenPayload,
) -> AppResult<String> {
    state.cancel_idle_timer();
    state.register_run(TITLE_GEN_RUN_ID);
    let result = title_gen_inner(&app, &state, &payload).await;
    state.finish_run(TITLE_GEN_RUN_ID);
    if state.session.lock().await.is_some() {
        arm_idle_timer(&app, &state, &payload.repo_id);
    }
    result
}

#[tauri::command]
pub fn cancel_run(
    state: tauri::State<'_, ChatState>,
    run_id: String,
) {
    state.request_cancel(&run_id);
}

#[tauri::command]
pub fn cancel_title_gen(state: tauri::State<'_, ChatState>) {
    state.request_cancel(TITLE_GEN_RUN_ID);
}

#[tauri::command]
pub async fn eject_session(
    app: AppHandle,
    state: tauri::State<'_, ChatState>,
    repo_id: String,
) -> AppResult<()> {
    state.cancel_idle_timer();
    if !state.evict(&app, &repo_id).await {
        emit_session_state(&app, false, &repo_id, false);
    }
    Ok(())
}

fn auto_eject_config_from(settings: &serde_json::Map<String, serde_json::Value>) -> (bool, Option<u64>) {
    let enabled = settings.get("autoEjectEnabled").and_then(|v| v.as_bool()).unwrap_or(true);
    let idle_ms = settings
        .get("autoEjectMinutes")
        .and_then(|v| v.as_f64())
        .filter(|m| m.is_finite() && *m > 0.0)
        .map(|minutes| ((minutes * 60_000.0) as u64).clamp(MIN_AUTO_EJECT_IDLE_MS, MAX_AUTO_EJECT_IDLE_MS));
    (enabled, idle_ms)
}

fn apply_auto_eject_config(
    app: &AppHandle,
    state: &ChatState,
    enabled: bool,
    idle_ms: Option<u64>,
) {
    let mut cfg = state.idle_config.lock().expect("idle_config mutex poisoned");
    cfg.enabled = enabled;
    if let Some(idle_ms) = idle_ms {
        cfg.idle_ms = idle_ms;
    }
    drop(cfg);
    if !enabled {
        state.cancel_idle_timer();
        return;
    }
    // An idle resident session gets the new timer now; a run in progress
    // re-arms it when it finishes.
    let Ok(_idle) = state.run_lock.try_lock() else {
        return;
    };
    let Ok(session) = state.session.try_lock() else {
        return;
    };
    if let Some(resident) = session.as_ref() {
        arm_idle_timer(app, state, &resident.repo_id);
    }
}

pub fn restore_auto_eject_config(app: &AppHandle) {
    let Ok(serde_json::Value::Object(settings)) = crate::storage::settings_load() else {
        return;
    };
    let (enabled, idle_ms) = auto_eject_config_from(&settings);
    apply_auto_eject_config(app, &app.state::<ChatState>(), enabled, idle_ms);
}

// Persisting and applying are one step, so the running timer never disagrees
// with what the next launch will read.
#[tauri::command]
pub fn set_auto_eject_config(
    app: AppHandle,
    state: tauri::State<'_, ChatState>,
    enabled: Option<bool>,
    minutes: Option<f64>,
) -> AppResult<()> {
    let mut patch = serde_json::Map::new();
    if let Some(enabled) = enabled {
        patch.insert("autoEjectEnabled".to_string(), enabled.into());
    }
    if let Some(minutes) = minutes {
        if !(minutes.is_finite() && minutes > 0.0) {
            return Err(crate::error::AppError::msg("autoEjectMinutes must be a positive number"));
        }
        patch.insert("autoEjectMinutes".to_string(), minutes.into());
    }
    let settings = crate::storage::settings_merge(patch)?;
    let (enabled, idle_ms) = auto_eject_config_from(&settings);
    apply_auto_eject_config(&app, &state, enabled, idle_ms);
    Ok(())
}

#[cfg(test)]
mod tests {
    use uzu::types::session::chat::ChatRole;

    #[test]
    fn auto_eject_defaults_to_enabled_with_no_interval_override() {
        let settings = serde_json::Map::new();
        assert_eq!(auto_eject_config_from(&settings), (true, None));
    }

    #[test]
    fn auto_eject_interval_is_clamped_and_bad_values_are_ignored() {
        let mut settings = serde_json::Map::new();
        settings.insert("autoEjectEnabled".to_string(), false.into());
        settings.insert("autoEjectMinutes".to_string(), 0.1.into());
        assert_eq!(auto_eject_config_from(&settings), (false, Some(MIN_AUTO_EJECT_IDLE_MS)));

        settings.insert("autoEjectMinutes".to_string(), 100_000.into());
        assert_eq!(auto_eject_config_from(&settings).1, Some(MAX_AUTO_EJECT_IDLE_MS));

        settings.insert("autoEjectMinutes".to_string(), (-3).into());
        assert_eq!(auto_eject_config_from(&settings).1, None);
    }

    use super::{payloads::test_message, *};

    #[test]
    fn continues_only_when_reasoning_prefix_matches() {
        let state = ChatState::default();
        let history = vec![
            test_message(ChatRole::User {}, "question", None),
            test_message(ChatRole::Assistant {}, "answer", Some("first path")),
        ];
        state.remember_history("model", history.clone(), Some(ReasoningEffort::Medium));

        let mut continued = history.clone();
        continued.push(test_message(ChatRole::User {}, "next", None));
        assert!(matches!(state.plan_run("model", &continued, Some(ReasoningEffort::Medium)), RunPlan::Continue { .. }));

        let mut changed = history;
        changed[1].reasoning_content = Some("second path".to_string());
        changed.push(test_message(ChatRole::User {}, "next", None));
        assert!(matches!(state.plan_run("model", &changed, Some(ReasoningEffort::Medium)), RunPlan::Replay));
    }
}
