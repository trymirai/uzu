use iocraft::prelude::*;

use crate::{
    cli::{
        components::{CommandInput, HistoryCell, HistoryCellType, Logo, Theme},
        flows::{ExitFlow, Flow, FlowEvent, FlowRegistry, ThemeFlow},
        helpers::SYMBOL_COMMAND,
    },
    engine::Engine,
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
        registry: FlowRegistry::default().register("theme", "Change the theme", || Box::new(ThemeFlow)).register(
            "exit",
            "Exit the CLI",
            || Box::new(ExitFlow),
        ),
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
        if state.read().flow.is_some() {
            state.write().flow = None;
        }
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

    let history_cell_components: Vec<AnyElement<'static>> =
        state.read().history.iter().cloned().map(|cell| element! { HistoryCell(r#type: Some(cell)) }.into()).collect();

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
                #(input_component)
            }
        }
    }
}
