use iocraft::prelude::*;
use shoji::types::model::Model;

use crate::{
    cli::{
        components::{CommandInput, HistoryCell, HistoryCellType, Logo, SelectedModel, Theme},
        flows::{ExitFlow, Flow, FlowEvent, FlowRegistry, ModelRegistriesFlow, ThemeFlow},
        helpers::SYMBOL_COMMAND,
    },
    engine::Engine,
    storage::types::DownloadState,
};

#[derive(Default, Props)]
pub struct ApplicationProps {
    pub engine: Option<Engine>,
}

pub struct ApplicationState {
    pub engine: Engine,
    pub theme: Theme,
    pub flow: Option<Box<dyn Flow>>,
    pub history: Vec<HistoryCellType>,
    pub registry: FlowRegistry,
    pub model: Option<Model>,
    pub model_download_state: Option<DownloadState>,
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
        model: None,
        model_download_state: None,
    });
    let (width, _) = hooks.use_terminal_size();

    let on_command = hooks.use_async_handler(move |text: String| async move {
        let Some(name) = text.strip_prefix(SYMBOL_COMMAND) else {
            return;
        };

        let mut state = state;
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
    });

    hooks.use_terminal_events(move |event| {
        let TerminalEvent::Key(KeyEvent {
            code: KeyCode::Esc,
            kind,
            ..
        }) = event
        else {
            return;
        };
        if kind == KeyEventKind::Release {
            return;
        }

        let mut state = state;
        let mut state = state.write();
        if state.flow.is_none() {
            return;
        }
        if matches!(state.history.last(), Some(HistoryCellType::Command { .. })) {
            state.history.push(HistoryCellType::CommandResult {
                result: "Cancelled".to_string(),
            });
        }
        state.flow = None;
    });

    let on_flow_event: Handler<FlowEvent> = Handler::from(move |event: FlowEvent| {
        let mut state = state;
        state.write().history.push(HistoryCellType::CommandResult {
            result: event.result,
        });
        state.write().flow = event.next_flow;
    });

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
        None => element! { CommandInput(on_submit: on_command) }.into(),
    };

    let history_cell_components: Vec<AnyElement<'static>> = state
        .read()
        .history
        .iter()
        .rev()
        .take(10)
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|r#type| element! { HistoryCell(r#type: Some(r#type)) }.into())
        .collect();

    let selected_model_component: AnyElement<'static> = match state.read().model.as_ref() {
        Some(model) => element! { SelectedModel(key: model.identifier.clone()) }.into(),
        None => element! { View }.into(),
    };

    element! {
        ContextProvider(value: Context::owned(state)) {
            View(
                flex_direction: FlexDirection::Column,
                width: width as u16,
                row_gap: state.read().theme.padding(),
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
                View(
                    flex_direction: FlexDirection::Column,
                    column_gap: 0,
                ) {
                    #(selected_model_component)
                    #(input_component)
                }
            }
        }
    }
}
