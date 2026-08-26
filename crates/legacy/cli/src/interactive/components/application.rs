use std::path::PathBuf;

use iocraft::prelude::*;
use shoji::types::{
    basic::{ReasoningEffort, SamplingParameters},
    model::Model,
};
use uzu::{
    engine::Engine,
    settings::Settings,
    storage::types::{DownloadPhase, DownloadState},
};

use crate::{
    common::thinking::{ThinkingPreference, ThinkingSupport},
    interactive::{
        APP_IDENTIFIER,
        components::{CommandInput, HistoryCell, HistoryCellType, Logo, Preferences, SelectedModel, Theme},
        flows::{AuthFlow, ExitFlow, Flow, FlowEvent, FlowRegistry, ModelRegistriesFlow, SettingsFlow, ThemeFlow},
        helpers::SYMBOL_COMMAND,
        model::resolve_model_id,
        sessions::{self, SessionState},
    },
};

const HISTORY_LIMIT: usize = 20;

#[derive(Default, Props)]
pub struct ApplicationProps {
    pub engine: Option<Engine>,
    pub settings: Option<Settings>,
    pub model: Option<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
}

pub struct ModelState {
    pub model: Model,
    pub download_state: DownloadState,
    pub session_state: Option<Box<dyn SessionState>>,
    pub thinking: ThinkingSupport,
    pub sampling_defaults: SamplingParameters,
    /// The thinking effort is baked into the prompt on the first turn and
    /// cannot be changed afterwards.
    pub thinking_locked: bool,
}

pub struct ApplicationState {
    preferences: Preferences,
    /// Session-scoped thinking override from the command line; never persisted.
    thinking_override: Option<ThinkingPreference>,
    pub engine: Engine,
    pub settings: Option<Settings>,
    pub flow: Option<Box<dyn Flow>>,
    pub history: Vec<HistoryCellType>,
    pub registry: FlowRegistry,
    pub model_state: Option<ModelState>,
}

impl ApplicationState {
    pub fn load_preferences() -> Result<Preferences, Box<dyn std::error::Error>> {
        Preferences::load(&Self::settings_file_path("config")?)
    }

    pub fn session_state(&self) -> Option<&dyn SessionState> {
        self.model_state.as_ref().and_then(|model_state| model_state.session_state.as_deref())
    }

    pub fn preferences(&self) -> &Preferences {
        &self.preferences
    }

    pub fn thinking(&self) -> ThinkingPreference {
        self.thinking_override.unwrap_or(self.preferences.thinking)
    }

    pub fn reasoning_effort_override(&self) -> Option<ReasoningEffort> {
        self.thinking_override.map(|preference| preference.level)
    }

    pub fn theme(&self) -> &Theme {
        &self.preferences.theme
    }

    pub fn set_preferences(
        &mut self,
        prefs: &Preferences,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if prefs.thinking != self.preferences.thinking {
            self.thinking_override = None;
        }
        self.preferences = prefs.clone();
        self.preferences.store(&Self::settings_file_path("config")?)
    }

    pub fn set_theme(
        &mut self,
        theme: Theme,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut preferences = self.preferences.clone();
        preferences.theme = theme;
        self.set_preferences(&preferences)
    }

    fn settings_file_path(name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        confy::get_configuration_file_path(APP_IDENTIFIER, name).map_err(Into::into)
    }
}

#[component]
pub fn Application(
    props: &ApplicationProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let engine = props.engine.clone().expect("Application requires an engine");

    let state = hooks.use_state(|| ApplicationState {
        engine,
        settings: props.settings.clone(),
        preferences: ApplicationState::load_preferences().unwrap_or_else(|error| {
            tracing::warn!("Unable to load or migrate CLI preferences: {error}");
            Preferences::default()
        }),
        thinking_override: props.reasoning_effort.map(|effort| ThinkingPreference {
            level: effort,
            enabled: effort != ReasoningEffort::Disabled,
        }),
        flow: None,
        history: Vec::new(),
        registry: FlowRegistry::default()
            .register("auth", "Add models from a specific provider", false, || Box::new(AuthFlow))
            .register("model", "Choose the model", false, || Box::new(ModelRegistriesFlow))
            .register("settings", "Configure thinking and sampling", true, || Box::new(SettingsFlow))
            .register("theme", "Choose the theme", false, || Box::new(ThemeFlow))
            .register("exit", "Exit the CLI", false, || Box::new(ExitFlow)),
        model_state: None,
    });
    let (width, _) = hooks.use_terminal_size();
    let requested_model = props.model.clone().or_else(|| state.read().preferences().selected_model_id.clone());

    hooks.use_future({
        let engine = state.read().engine.clone();
        let mut state = state;
        async move {
            let initial_model = match requested_model {
                Some(model) => match resolve_model_id(&engine, model).await {
                    Ok(model) => model,
                    Err(error) => {
                        state.write().history.push(HistoryCellType::CommandResult {
                            result: format!("Failed to resolve model: {error}"),
                        });
                        state.write().flow = Some(Box::new(ModelRegistriesFlow));
                        return;
                    },
                },
                None => None,
            };

            let Some(identifier) = initial_model else {
                state.write().flow = Some(Box::new(ModelRegistriesFlow));
                return;
            };
            match engine.model(identifier.clone()).await {
                Ok(Some(model)) => {
                    let model_exists = model.is_downloadable()
                        || !model.is_local()
                        || matches!(
                            engine.model_path(&model).await,
                            Some(path) if std::path::Path::new(&path).exists()
                        );
                    if !model_exists {
                        state.write().flow = Some(Box::new(ModelRegistriesFlow));
                        return;
                    }
                    state.write().model_state = Some(ModelState {
                        model,
                        download_state: DownloadState::not_downloaded(0),
                        session_state: None,
                        thinking: ThinkingSupport::default(),
                        sampling_defaults: SamplingParameters::default(),
                        thinking_locked: false,
                    });
                },
                Ok(None) => {
                    state.write().history.push(HistoryCellType::CommandResult {
                        result: format!("Unknown model: {}", identifier),
                    });
                    state.write().flow = Some(Box::new(ModelRegistriesFlow));
                },
                Err(error) => {
                    state.write().history.push(HistoryCellType::CommandResult {
                        result: format!("Failed to load model {}: {}", identifier, error),
                    });
                    state.write().flow = Some(Box::new(ModelRegistriesFlow));
                },
            }
        }
    });

    let on_command = hooks.use_async_handler(move |text: String| async move {
        let mut state = state;

        if let Some(name) = text.strip_prefix(SYMBOL_COMMAND) {
            state.write().history.push(HistoryCellType::Command {
                name: name.to_string(),
            });
            let registry = state.read().registry.clone();
            let model_loaded = state.read().model_state.is_some();
            match registry.command(name) {
                Some(command) if command.requires_model && !model_loaded => {
                    state.write().history.push(HistoryCellType::CommandResult {
                        result: "Load a model first to open settings".to_string(),
                    })
                },
                Some(command) => state.write().flow = Some((command.factory)()),
                None => state.write().history.push(HistoryCellType::CommandResult {
                    result: format!("Unknown command: /{}", name),
                }),
            }
            return;
        }

        state.write().history.push(HistoryCellType::Request {
            text: text.clone(),
        });

        let model_with_download_state = state
            .read()
            .model_state
            .as_ref()
            .map(|model_state| (model_state.model.clone(), model_state.download_state.clone()));
        let (model, download_state) = match model_with_download_state {
            Some(pair) => pair,
            None => {
                state.write().history.push(HistoryCellType::CommandResult {
                    result: "No model is selected".to_string(),
                });
                return;
            },
        };

        if model.is_downloadable() && !matches!(download_state.phase, DownloadPhase::Downloaded {}) {
            state.write().history.push(HistoryCellType::CommandResult {
                result: "Model is not downloaded".to_string(),
            });
            let engine = state.read().engine.clone();
            let downloader = engine.downloader(&model);
            let _ = downloader.resume().await;
            return;
        }

        if model.is_chat_capable() {
            let has_running_session = state.read().session_state().is_some_and(SessionState::is_busy);
            if has_running_session {
                return;
            }
            let Some(session) = sessions::chat::ensure_session(state, &model).await else {
                return;
            };
            sessions::chat::run_session(state, session, text).await;
        } else if model.is_classification_capable() {
            let has_running_session = state.read().session_state().is_some_and(SessionState::is_busy);
            if has_running_session {
                return;
            }
            let Some(session) = sessions::classification::ensure_session(state, &model).await else {
                return;
            };
            sessions::classification::run_session(state, session, text).await;
        } else if model.is_text_to_speech_capable() {
            let has_running_session = state.read().session_state().is_some_and(SessionState::is_busy);
            if has_running_session {
                return;
            }
            let Some(runtime) = sessions::text_to_speech::ensure_session(state, &model).await else {
                return;
            };
            sessions::text_to_speech::run_session(state, runtime, text).await;
        } else {
            state.write().history.push(HistoryCellType::CommandResult {
                result: "Model is not supported yet".to_string(),
            });
        }
    });

    hooks.use_terminal_events(move |event| {
        let TerminalEvent::Key(KeyEvent {
            code,
            kind,
            modifiers,
            ..
        }) = event
        else {
            return;
        };
        if kind == KeyEventKind::Release {
            return;
        }

        let is_escape = matches!(code, KeyCode::Esc);
        let is_ctrl_c = matches!(code, KeyCode::Char('c')) && modifiers.contains(KeyModifiers::CONTROL);
        if !is_escape && !is_ctrl_c {
            return;
        }

        let mut state = state;
        let mut state = state.write();

        let consumed_by_session = if let Some(session_state) = state.session_state() {
            if is_escape {
                session_state.interrupt() || session_state.is_busy()
            } else {
                session_state.is_busy()
            }
        } else {
            false
        };
        if consumed_by_session {
            return;
        }

        if state.flow.is_some() {
            if is_escape {
                if matches!(state.history.last(), Some(HistoryCellType::Command { .. })) {
                    state.history.push(HistoryCellType::CommandResult {
                        result: "Cancelled".to_string(),
                    });
                }
                state.flow = None;
            }
            return;
        }

        state.flow = Some(Box::new(ExitFlow));
    });

    let on_flow_event: Handler<FlowEvent> = Handler::from(move |event: FlowEvent| {
        let mut state = state;
        state.write().history.push(HistoryCellType::CommandResult {
            result: event.result,
        });
        state.write().flow = event.next_flow;
    });

    let input_disabled = state.read().session_state().is_some_and(SessionState::is_busy);
    let input_component: AnyElement<'static> = match state.read().flow.as_ref() {
        Some(flow) => {
            let flow_component = flow.render(on_flow_event);
            element! {
                View(flex_direction: FlexDirection::Column) {
                    View(
                        width: 100pct,
                        height: 1u16,
                        border_style: BorderStyle::Single,
                        border_color: state.read().theme().accent_color,
                        border_edges: Some(Edges::Top),
                    )
                    #(flow_component)
                }
            }
            .into()
        },
        None => element! { CommandInput(disabled: input_disabled, on_submit: on_command) }.into(),
    };

    let history_cell_components: Vec<AnyElement<'static>> = state
        .read()
        .history
        .iter()
        .rev()
        .take(HISTORY_LIMIT)
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|r#type| element! { HistoryCell(r#type: Some(r#type)) }.into())
        .collect();

    let pending_reply = {
        let state = state.read();
        state.session_state().and_then(SessionState::pending_history_cell)
    };
    let pending_reply_component: AnyElement<'static> = match pending_reply {
        Some(history_cell) => element! {
            HistoryCell(r#type: Some(history_cell), live: true)
        }
        .into(),
        None => element! { View }.into(),
    };

    let selected_model_component: AnyElement<'static> = match state.read().model_state.as_ref() {
        Some(model_state) => element! { SelectedModel(key: model_state.model.identifier.clone()) }.into(),
        None => element! { View }.into(),
    };

    element! {
        ContextProvider(value: Context::owned(state)) {
            View(
                flex_direction: FlexDirection::Column,
                width: width,
            ) {
                View(
                    padding_left: state.read().theme().padding(),
                    padding_right: state.read().theme().padding(),
                ) {
                    Logo
                }
                View(
                    flex_direction: FlexDirection::Column
                ) {
                    #(history_cell_components.into_iter())
                }
                #(pending_reply_component)
                View(
                    flex_direction: FlexDirection::Column,
                    column_gap: 0,
                ) {
                    View(height: state.read().theme().padding())
                    #(selected_model_component)
                    #(input_component)
                }
            }
        }
    }
}
