use tauri::{AppHandle, Manager};
use uzu::{
    engine::Engine,
    session::{
        chat::{ChatInstance, ChatSession},
        tool::uzu_tool_function,
    },
    types::session::chat::ChatConfig,
};

use super::{
    ChatState, ResidentSession,
    chart::show_chart,
    naming::ChatNaming,
    payloads::RunStreamPayload,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct RequestedTools {
    pub chat_naming: Option<bool>,
    pub date_time: Option<bool>,
    pub chart: Option<bool>,
}

impl RequestedTools {
    pub(super) const ALL_OFF: Self = Self {
        chat_naming: Some(false),
        date_time: Some(false),
        chart: Some(false),
    };
}

impl From<&RunStreamPayload> for RequestedTools {
    fn from(payload: &RunStreamPayload) -> Self {
        Self {
            chat_naming: payload.model_chat_naming_enabled,
            date_time: payload.date_time_tool_enabled,
            chart: payload.chart_tool_enabled,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct EnabledTools {
    pub chat_naming: bool,
    pub date_time: bool,
    pub chart: bool,
}

impl EnabledTools {
    fn resolve(
        requested: RequestedTools,
        model_size: Option<i64>,
        settings: &serde_json::Value,
    ) -> Self {
        Self {
            chat_naming: chat_naming_enabled(resolve_tool_enabled(requested.chat_naming, model_size), settings),
            date_time: resolve_tool_enabled(requested.date_time, model_size),
            chart: resolve_tool_enabled(requested.chart, model_size),
        }
    }

    pub(super) fn any(self) -> bool {
        self.chat_naming || self.date_time || self.chart
    }
}

pub(super) struct ReadySession {
    pub session: ChatSession,
    pub support: ReasoningSupport,
    pub naming: Option<ChatNaming>,
    pub tools: EnabledTools,
    pub identifier: String,
}

pub(super) async fn ensure_session(
    app: &AppHandle,
    state: &ChatState,
    repo_id: &str,
    requested: RequestedTools,
) -> AppResult<ReadySession> {
    let model = find_chat_model(repo_id).await?;
    let model_size = model.properties.as_ref().map(|properties| properties.size);
    let settings = crate::storage::settings_load()?;
    let tools = EnabledTools::resolve(requested, model_size, &settings);
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
        if resident.tools == tools {
            return Ok(resident.ready());
        }
        // Tools survive reset(). Drop the old KV state before creating the
        // replacement session, keeping the loaded model weights in the instance.
        crate::logger::info(
            "chat:session:replaced",
            Some(serde_json::json!({ "repoId": repo_id, "reason": "toolsChanged" })),
        );
        let resident = guard.take().expect("resident session");
        drop(resident.session);
        state.forget_history();
        let (session, naming) = match create_session(&engine, &resident.instance, tools).await {
            Ok(created) => created,
            Err(error) => {
                emit_session_state(app, false, repo_id, false);
                return Err(error);
            },
        };
        let resident = ResidentSession {
            session,
            naming,
            tools,
            ..resident
        };
        let ready = resident.ready();
        *guard = Some(resident);
        return Ok(ready);
    }
    let support = reasoning_support(&model);
    *guard = None;
    state.forget_history();
    // Loading events only on the real load path: emitting them on session reuse
    // would flash the spinner and the resident-model chip.
    emit_session_loading(app, LoadingStatus::Start, repo_id, None);
    let config = ChatConfig::create();
    let loaded = async {
        let instance = engine.chat_instance(model.clone(), config).await?;
        let (session, naming) = create_session(&engine, &instance, tools).await?;
        Ok::<_, AppError>((instance, session, naming))
    }
    .await;
    let (instance, session, naming) = match loaded {
        Ok(loaded) => loaded,
        Err(error) => {
            emit_session_loading(app, LoadingStatus::Error, repo_id, Some(error.to_string()));
            return Err(error);
        },
    };
    let resident = ResidentSession {
        repo_id: repo_id.to_string(),
        identifier: model.identifier.clone(),
        session,
        instance,
        tools,
        naming,
        support,
    };
    let ready = resident.ready();
    *guard = Some(resident);
    emit_session_state(app, true, repo_id, false);
    Ok(ready)
}

impl ResidentSession {
    fn ready(&self) -> ReadySession {
        ReadySession {
            session: self.session.clone(),
            support: self.support.clone(),
            naming: self.naming.clone(),
            tools: self.tools,
            identifier: self.identifier.clone(),
        }
    }
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
    tools: EnabledTools,
) -> AppResult<(ChatSession, Option<ChatNaming>)> {
    let mut session = engine.chat_with_instance(instance).await?;
    let supports_tools = session.supports_tool_calls().await;
    if tools.date_time && supports_tools {
        session.add_tool(get_current_date_time).await?;
    }
    if tools.chart && supports_tools {
        session.add_tool(show_chart).await?;
    }
    let naming = if tools.chat_naming && supports_tools {
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
    let now = jiff::Zoned::now();
    format!(
        "Current UTC date and time: {}.\nDevice local date and time: {}.",
        now.timestamp().strftime("%A, %-d %B %Y at %H:%M:%S (UTC%:z)"),
        now.strftime("%A, %-d %B %Y at %H:%M:%S (UTC%:z)"),
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
        let before = jiff::Zoned::now();
        let result = tool.execute(serde_json::json!({}).into()).await.unwrap();
        let after = jiff::Zoned::now();
        let value: serde_json::Value = result.try_into().unwrap();
        let (utc, local) = value.as_str().unwrap().split_once('\n').unwrap();
        let utc = jiff::Zoned::strptime("Current UTC date and time: %A, %-d %B %Y at %H:%M:%S (UTC%:z).", utc).unwrap();
        let local =
            jiff::Zoned::strptime("Device local date and time: %A, %-d %B %Y at %H:%M:%S (UTC%:z).", local).unwrap();
        assert_eq!(utc.timestamp(), local.timestamp());
        assert!(
            utc.timestamp().as_second() >= before.timestamp().as_second()
                && utc.timestamp().as_second() <= after.timestamp().as_second()
        );
        assert_eq!(utc.offset(), jiff::tz::Offset::UTC);
        assert_eq!(local.offset(), local.with_time_zone(jiff::tz::TimeZone::system()).offset());
    }
}
