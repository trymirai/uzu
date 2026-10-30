use iocraft::prelude::*;
use shoji::types::model::ModelSpecialization;
use unicode_width::UnicodeWidthStr;
use uzu::storage::DownloadPhase;

use crate::interactive::{
    components::{ApplicationState, InputType, Selector, SelectorItem, SelectorStyle, TextInput, TextInputFocus},
    flows::Command,
    helpers::{
        HINT_COMMANDS, HINT_DECISION_QUESTION_EXAMPLES, HINT_DECISION_QUESTION_NOTES, HINT_DECISION_QUESTION_TITLE,
        HINT_DECISION_STATE_EXAMPLES, HINT_DECISION_STATE_NOTES, HINT_DECISION_STATE_TITLE, HINT_SEND,
        HINT_STORAGE_DELETE, HINT_STORAGE_PAUSE_RESUME, HintExample, SYMBOL_COMMAND, SYMBOL_INPUT,
    },
    sessions::decision::is_question_mode,
};

const SAFE_PADDING: u16 = 1;

#[derive(Default, Props)]
pub struct CommandInputProps {
    pub disabled: bool,
    pub on_submit: Handler<String>,
}

#[component]
pub fn CommandInput(
    props: &CommandInputProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let disabled = props.disabled;
    let on_submit = props.on_submit.clone();

    let state = *hooks.use_context::<State<ApplicationState>>();
    let mut input = hooks.use_state(String::new);
    let (width, _) = hooks.use_terminal_size();

    let input_text = input.read().clone();
    let valid = valid_commands(&input_text, state);
    let focus = if disabled {
        TextInputFocus::Disabled
    } else if valid.is_empty() {
        TextInputFocus::Full
    } else {
        TextInputFocus::Minimal
    };

    let padding = state.read().theme().padding();
    let accent_color = state.read().theme().accent_color;
    let subtitle_color = state.read().theme().subtitle_color;
    let overlay_color = state.read().theme().overlay_color();
    let maximal_width = width
        .saturating_sub(UnicodeWidthStr::width(SYMBOL_INPUT) as u16)
        .saturating_sub(padding)
        .saturating_sub(SAFE_PADDING);

    element! {
        View(
            flex_direction: FlexDirection::Column,
            background_color: if disabled { Some(overlay_color) } else { None },
        ) {
            View(
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::FlexStart,
                width: 100pct,
                column_gap: padding,
                border_color: subtitle_color,
                border_style: BorderStyle::Single,
                border_edges: Some(Edges::Top | Edges::Bottom),
            ) {
                Text(content: SYMBOL_INPUT, color: accent_color)
                TextInput(
                    maximal_width: maximal_width,
                    focus: focus,
                    r#type: InputType::Text,
                    on_change: move |text: String| input.set(text),
                    on_submit: on_submit.clone(),
                )
            }
            #(hint_component(valid, state, on_submit))
        }
    }
}

fn valid_commands(
    input: &str,
    state: State<ApplicationState>,
) -> Vec<Command> {
    let Some(command_name) = input.strip_prefix(SYMBOL_COMMAND) else {
        return Vec::new();
    };
    let model_loaded = state.read().model_state.is_some();
    state
        .read()
        .registry
        .commands()
        .into_iter()
        .filter(|command| command.name.starts_with(command_name))
        .filter(|command| !command.requires_model || model_loaded)
        .collect()
}

fn hint_component(
    commands: Vec<Command>,
    state: State<ApplicationState>,
    on_submit: Handler<String>,
) -> AnyElement<'static> {
    let mut hints = Vec::new();
    match state.read().model_state.as_ref() {
        Some(model_state) if model_state.model.is_downloadable() => {
            if !matches!(model_state.download_state.phase, DownloadPhase::Downloaded {}) {
                hints.push(HINT_STORAGE_PAUSE_RESUME.to_string());
            } else {
                hints.push(HINT_SEND.to_string());
            }
            hints.push(HINT_STORAGE_DELETE.to_string());
        },
        _ => {
            hints.push(HINT_SEND.to_string());
        },
    }
    hints.push(HINT_COMMANDS.to_string());
    let is_decision_model = state
        .read()
        .model_state
        .as_ref()
        .is_some_and(|model_state| model_state.model.specializations.contains(&ModelSpecialization::Decision {}));

    if commands.is_empty() {
        return element! {
            View(flex_direction: FlexDirection::Column) {
                Text(
                    content: hints.join("\n"),
                    color: state.read().theme().subtitle_color,
                )
                #(is_decision_model.then(|| if is_question_mode(&state.read()) {
                    decision_guide_component(
                        state,
                        HINT_DECISION_QUESTION_TITLE,
                        HINT_DECISION_QUESTION_EXAMPLES,
                        HINT_DECISION_QUESTION_NOTES,
                    )
                } else {
                    decision_guide_component(
                        state,
                        HINT_DECISION_STATE_TITLE,
                        HINT_DECISION_STATE_EXAMPLES,
                        HINT_DECISION_STATE_NOTES,
                    )
                }))
            }
        }
        .into();
    }

    let items: Vec<SelectorItem> = commands
        .iter()
        .map(|command| SelectorItem {
            title: command.name.clone(),
            description: Some(command.description.clone()),
            color: None,
        })
        .collect();
    let maximal_height = (items.len() as u16).min(5);
    let accent_color = state.read().theme().accent_color;
    let subtitle_color = state.read().theme().subtitle_color;
    let columns_padding = state.read().theme().padding_wide();

    element! {
        Selector(
            items: items,
            style: SelectorStyle::Plain,
            maximal_height: maximal_height,
            accent_color: accent_color,
            subtitle_color: subtitle_color,
            columns_padding: columns_padding,
            on_submit: move |index: usize| {
                if let Some(command) = commands.get(index) {
                    on_submit(format!("{}{}", SYMBOL_COMMAND, command.name));
                }
            },
        )
    }
    .into()
}

fn decision_guide_component(
    state: State<ApplicationState>,
    title: &'static str,
    examples: &'static [HintExample],
    notes: &'static [&'static str],
) -> AnyElement<'static> {
    let theme = state.read().theme().clone();
    let label_width = examples.iter().map(|example| UnicodeWidthStr::width(example.label) as u16).max().unwrap_or(0);
    let lines_width = examples
        .iter()
        .flat_map(|example| example.lines.iter())
        .map(|line| UnicodeWidthStr::width(*line) as u16)
        .max()
        .unwrap_or(0);
    let rows: Vec<AnyElement<'static>> = examples
        .iter()
        .map(|example| {
            element! {
                View(flex_direction: FlexDirection::Row, column_gap: theme.padding_wide()) {
                    View(width: label_width) {
                        Text(content: example.label, color: theme.subtitle_color)
                    }
                    View(width: lines_width) {
                        Text(content: example.lines.join("\n"))
                    }
                    Text(content: example.note, color: theme.subtitle_color)
                }
            }
            .into()
        })
        .collect();

    element! {
        View(
            width: 100pct,
            flex_direction: FlexDirection::Column,
            row_gap: theme.padding(),
            border_style: BorderStyle::Single,
            border_color: theme.subtitle_color,
            padding_left: theme.padding(),
            padding_right: theme.padding(),
        ) {
            Text(content: title, weight: Weight::Bold)
            #(rows.into_iter())
            Text(content: notes.join("\n"), color: theme.subtitle_color)
        }
    }
    .into()
}
