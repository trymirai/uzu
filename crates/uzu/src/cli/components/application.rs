use iocraft::prelude::*;
use shoji::types::model::Model;

use crate::{
    cli::{
        components::{CommandInput, HistoryCell, HistoryCellType, Logo, SelectedModel, Theme},
        flows::{ExitFlow, Flow, FlowEvent, FlowRegistry, ModelRegistriesFlow, ThemeFlow},
        helpers::SYMBOL_COMMAND,
        sessions::{self, SessionState},
    },
    engine::Engine,
    storage::types::{DownloadPhase, DownloadState},
};

const HISTORY_LIMIT: usize = 20;

#[derive(Default, Props)]
pub struct ApplicationProps {
    pub engine: Option<Engine>,
}

pub struct ModelState {
    pub model: Model,
    pub download_state: DownloadState,
    pub session_state: Option<Box<dyn SessionState>>,
}

pub struct ApplicationState {
    pub engine: Engine,
    pub theme: Theme,
    pub flow: Option<Box<dyn Flow>>,
    pub history: Vec<HistoryCellType>,
    pub registry: FlowRegistry,
    pub model_state: Option<ModelState>,
}

impl ApplicationState {
    pub fn session_state(&self) -> Option<&dyn SessionState> {
        self.model_state.as_ref().and_then(|model_state| model_state.session_state.as_deref())
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
        theme: Theme::default(),
        flow: None,
        history: Vec::new(),
        registry: FlowRegistry::default()
            .register("theme", "Choose the theme", || Box::new(ThemeFlow))
            .register("model", "Choose the model", || Box::new(ModelRegistriesFlow))
            .register("exit", "Exit the CLI", || Box::new(ExitFlow)),
        model_state: None,
    });
    let (width, _) = hooks.use_terminal_size();

    let on_command = hooks.use_async_handler(move |text: String| async move {
        let mut state = state;

        if let Some(name) = text.strip_prefix(SYMBOL_COMMAND) {
            state.write().history.push(HistoryCellType::Command {
                name: name.to_string(),
            });
            let registry = state.read().registry.clone();
            match registry.create(name) {
                Some(flow) => state.write().flow = Some(flow),
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
                session_state.cancel() || session_state.is_busy()
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
                        border_color: state.read().theme.accent_color,
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
            HistoryCell(r#type: Some(history_cell))
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
                width: width as u16,
            ) {
                View(
                    padding_left: state.read().theme.padding(),
                    padding_right: state.read().theme.padding(),
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
                    View(height: state.read().theme.padding())
                    #(selected_model_component)
                    #(input_component)
                }
            }
        }
    }
}
