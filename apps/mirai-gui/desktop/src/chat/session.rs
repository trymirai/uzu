use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

use tauri::{AppHandle, Emitter, Manager};
use uzu::{
    engine::{Engine, EngineError},
    session::{
        chat::{ChatInstance, ChatSession, ChatSessionError},
        tool::uzu_tool_function,
    },
    types::{basic::BackendError, model::Model, session::chat::ChatConfig},
};

use super::{
    ChatState, ResidentSession,
    chart::show_chart,
    naming::ChatNaming,
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
    model_chat_naming_enabled: Option<bool>,
    date_time_tool_enabled: Option<bool>,
    chart_tool_enabled: Option<bool>,
) -> AppResult<(ChatSession, ReasoningSupport, Option<ChatNaming>)> {
    let model = find_chat_model(repo_id).await?;
    let model_size = model.properties.as_ref().map(|properties| properties.size);
    let model_chat_naming_enabled = chat_naming_enabled(
        resolve_tool_enabled(model_chat_naming_enabled, model_size),
        &crate::storage::settings_load()?,
    );
    let date_time_tool_enabled = resolve_tool_enabled(date_time_tool_enabled, model_size);
    let chart_tool_enabled = resolve_tool_enabled(chart_tool_enabled, model_size);
    let engine = engine().await?;
    if model.is_downloadable()
        && !matches!(engine.ready_download_state(&model).await?.phase, uzu::storage::DownloadPhase::Downloaded {})
    {
        return Err(AppError::msg("Model is not downloaded"));
    }
    let mut guard = state.session.lock().await;
    if let Some(resident) = guard.as_ref()
        && resident.identifier == model.identifier
    {
        if resident.model_chat_naming_enabled == model_chat_naming_enabled
            && resident.date_time_tool_enabled == date_time_tool_enabled
            && resident.chart_tool_enabled == chart_tool_enabled
        {
            return Ok((resident.session.clone(), resident.support.clone(), resident.naming.clone()));
        }
        // Tools survive reset(). Drop the old KV state before creating the
        // replacement session, keeping the loaded model weights in the instance.
        let resident = guard.take().expect("resident session");
        drop(resident.session);
        state.forget_history();
        let (session, naming) = match create_session(
            &engine,
            &resident.instance,
            model_chat_naming_enabled,
            date_time_tool_enabled,
            chart_tool_enabled,
        )
        .await
        {
            Ok(created) => created,
            Err(error) => {
                emit_session_state(app, false, repo_id, false);
                return Err(error);
            },
        };
        let support = resident.support.clone();
        *guard = Some(ResidentSession {
            session: session.clone(),
            naming: naming.clone(),
            model_chat_naming_enabled,
            date_time_tool_enabled,
            chart_tool_enabled,
            ..resident
        });
        return Ok((session, support, naming));
    }
    let support = reasoning_support(&model);
    *guard = None;
    state.forget_history();
    // Loading events only on the real load path: emitting them on session reuse
    // would flash the spinner and the resident-model chip.
    emit_session_loading(app, LoadingStatus::Start, repo_id, None);
    let config = ChatConfig::create();
    let loaded = async {
        let instance = engine.chat_instance(model.clone(), config).await.map_err(load_error)?;
        if let Some(notice) = instance.speculation_notice() {
            crate::logger::info(
                "session:speculation-off",
                Some(serde_json::json!({ "repoId": repo_id, "notice": notice })),
            );
        }
        let (session, naming) =
            create_session(&engine, &instance, model_chat_naming_enabled, date_time_tool_enabled, chart_tool_enabled)
                .await?;
        Ok::<_, AppError>((instance, session, naming))
    }
    .await;
    let (instance, session, naming) = match loaded {
        Ok(loaded) => loaded,
        Err(error) => {
            forget_memory_hints(app);
            emit_session_loading(app, LoadingStatus::Error, repo_id, Some(error.to_string()));
            return Err(error);
        },
    };
    MEMORY_HINTS.lock().expect("memory hints mutex poisoned").insert(model.identifier.clone(), None);
    *guard = Some(ResidentSession {
        repo_id: repo_id.to_string(),
        identifier: model.identifier.clone(),
        session: session.clone(),
        instance,
        model_chat_naming_enabled,
        date_time_tool_enabled,
        chart_tool_enabled,
        naming: naming.clone(),
        support: support.clone(),
    });
    emit_session_state(app, true, repo_id, false);
    Ok((session, support, naming))
}

// The backend refuses a model too large for the device with an error that explains it; the chat shows the explanation
// alone.
fn load_error(error: EngineError) -> AppError {
    match error {
        EngineError::ChatSession(ChatSessionError::Backend {
            error: BackendError::Other {
                message,
            },
        }) => AppError::msg(message),
        error => error.into(),
    }
}

static MEMORY_HINTS: LazyLock<Mutex<HashMap<String, Option<u64>>>> = LazyLock::new(Default::default);

/// The device memory free when the downloaded model `identifier` was found too large for it; the outer `None` until
/// it is checked.
pub fn memory_hint(identifier: &str) -> Option<Option<u64>> {
    MEMORY_HINTS.lock().expect("memory hints mutex poisoned").get(identifier).copied()
}

/// Forgets every hint, as the memory free changes when a model unloads or fails to load, and has the models listed anew.
pub fn forget_memory_hints(app: &AppHandle) {
    MEMORY_HINTS.lock().expect("memory hints mutex poisoned").clear();
    let _ = app.emit("models-changed", ());
}

/// Marks the downloaded `models` checked, then checks them against the device memory free, queried once, and announces
/// the hints. Waits out a load, and skips the check while a session is loaded, whose memory counts as used.
pub fn refresh_memory_hints(
    app: AppHandle,
    engine: Engine,
    models: Vec<Model>,
) {
    MEMORY_HINTS
        .lock()
        .expect("memory hints mutex poisoned")
        .extend(models.iter().map(|model| (model.identifier.clone(), None)));
    tauri::async_runtime::spawn(async move {
        if app.state::<ChatState>().session.lock().await.is_some() {
            return;
        }
        let warn = |error: EngineError| {
            crate::logger::warn("models:memory-hints", Some(serde_json::json!({ "error": error.to_string() })));
        };
        let available = match engine.available_memory().await {
            Ok(Some(available)) => available,
            Ok(None) => return,
            Err(error) => return warn(error),
        };
        let mut changed = false;
        for model in models {
            match engine.memory_check(&model).await {
                Ok(check) => {
                    if let Some(shortfall) = check.and_then(|check| check(available)) {
                        MEMORY_HINTS
                            .lock()
                            .expect("memory hints mutex poisoned")
                            .insert(model.identifier, Some(shortfall.available_bytes));
                        changed = true;
                    }
                },
                Err(error) => warn(error),
            }
        }
        if changed {
            let _ = app.emit("models-changed", ());
        }
    });
}

fn resolve_tool_enabled(
    requested: Option<bool>,
    model_size: Option<i64>,
) -> bool {
    requested.unwrap_or_else(|| model_size.is_none_or(|size| size >= 2_000_000_000))
}

fn chat_naming_enabled(
    requested: bool,
    settings: &serde_json::Value,
) -> bool {
    requested && settings.get("modelChatNamingEnabled").and_then(serde_json::Value::as_bool) != Some(false)
}

async fn create_session(
    engine: &Engine,
    instance: &ChatInstance,
    model_chat_naming_enabled: bool,
    date_time_tool_enabled: bool,
    chart_tool_enabled: bool,
) -> AppResult<(ChatSession, Option<ChatNaming>)> {
    let mut session = engine.chat_with_instance(instance).await?;
    let supports_tools = session.supports_tool_calls().await;
    if date_time_tool_enabled && supports_tools {
        session.add_tool(get_current_date_time).await?;
    }
    if chart_tool_enabled && supports_tools {
        session.add_tool(show_chart).await?;
    }
    let naming = if model_chat_naming_enabled && supports_tools {
        let naming = ChatNaming::default();
        session.add_tool(naming.tool()).await?;
        Some(naming)
    } else {
        None
    };
    Ok((session, naming))
}

/// Returns the current date and time in UTC and the device's local time, with explicit UTC offsets.
#[uzu_tool_function]
fn get_current_date_time() -> String {
    let now = chrono::Local::now();
    format!(
        "Current UTC date and time: {}.\nDevice local date and time: {}.",
        now.with_timezone(&chrono::Utc).format("%A, %-d %B %Y at %H:%M:%S (UTC%:z)"),
        now.format("%A, %-d %B %Y at %H:%M:%S (UTC%:z)"),
    )
}

#[cfg(test)]
mod tests {
    use uzu::session::tool::func_def::ToolDescriptor;

    use super::*;

    #[test]
    fn tool_defaults_follow_model_size_without_overriding_explicit_choices() {
        for (size, default_enabled) in
            [(Some(1_999_999_999), false), (Some(2_000_000_000), true), (Some(27_000_000_000), true), (None, true)]
        {
            assert_eq!(resolve_tool_enabled(None, size), default_enabled);
            assert!(resolve_tool_enabled(Some(true), size));
            assert!(!resolve_tool_enabled(Some(false), size));
        }
    }

    #[test]
    fn global_naming_switch_overrides_request_without_overriding_per_model_disable() {
        let mut settings = serde_json::json!({
            "modelChatNamingEnabled": false,
            "modelParams": {"test/model": {"modelChatNamingEnabled": true}},
        });
        let explicit_small_model_opt_in = resolve_tool_enabled(Some(true), Some(1_200_000_000));
        assert!(!chat_naming_enabled(explicit_small_model_opt_in, &settings));

        settings["modelChatNamingEnabled"] = true.into();
        assert!(chat_naming_enabled(explicit_small_model_opt_in, &settings));
        assert!(!chat_naming_enabled(false, &settings));
        assert!(chat_naming_enabled(true, &serde_json::json!({})));
    }

    #[tokio::test]
    async fn date_time_tool_returns_the_same_current_instant_in_utc_and_local_time() {
        let tool: ToolDescriptor = get_current_date_time.into();
        assert_eq!(tool.name, "get_current_date_time");
        let before = chrono::Local::now();
        let result = tool.execute(serde_json::json!({}).into()).await.unwrap();
        let after = chrono::Local::now();
        let value: serde_json::Value = result.try_into().unwrap();
        let (utc, local) = value.as_str().unwrap().split_once('\n').unwrap();
        let utc =
            chrono::DateTime::parse_from_str(utc, "Current UTC date and time: %A, %-d %B %Y at %H:%M:%S (UTC%:z).")
                .unwrap();
        let local =
            chrono::DateTime::parse_from_str(local, "Device local date and time: %A, %-d %B %Y at %H:%M:%S (UTC%:z).")
                .unwrap();
        assert_eq!(utc, local);
        assert!(utc.timestamp() >= before.timestamp() && utc.timestamp() <= after.timestamp());
        assert_eq!(utc.offset().local_minus_utc(), 0);
        assert_eq!(local.offset(), local.with_timezone(&chrono::Local).offset());
    }
}
