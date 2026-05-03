use iocraft::prelude::*;
use unicode_width::UnicodeWidthStr;

use crate::{
    cli::{
        components::{ApplicationState, Selector, SelectorItem, SelectorStyle, TextInput, TextInputFocus},
        flows::Command,
        helpers::{
            HINT_COMMANDS, HINT_SEND, HINT_STORAGE_DELETE, HINT_STORAGE_PAUSE_RESUME, SYMBOL_COMMAND, SYMBOL_INPUT,
        },
    },
    storage::types::DownloadPhase,
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
    let mut input = hooks.use_state(|| String::new());
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

    let padding = state.read().theme.padding();
    let accent_color = state.read().theme.accent_color;
    let subtitle_color = state.read().theme.subtitle_color;
    let overlay_color = state.read().theme.overlay_color();
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
    state.read().registry.commands().into_iter().filter(|command| command.name.starts_with(command_name)).collect()
}

fn hint_component(
    commands: Vec<Command>,
    state: State<ApplicationState>,
    on_submit: Handler<String>,
) -> AnyElement<'static> {
    let mut hints = Vec::new();
    match state.read().model_state.as_ref() {
        Some(model_state) => {
            if model_state.model.is_downloadable() {
                if !matches!(model_state.download_state.phase, DownloadPhase::Downloaded {}) {
                    hints.push(HINT_STORAGE_PAUSE_RESUME.to_string());
                } else {
                    hints.push(HINT_SEND.to_string());
                }
                hints.push(HINT_STORAGE_DELETE.to_string());
            } else {
                hints.push(HINT_SEND.to_string());
            }
        },
        None => {
            hints.push(HINT_SEND.to_string());
        },
    }
    hints.push(HINT_COMMANDS.to_string());

    if commands.is_empty() {
        return element! {
            Text(
                content: hints.join("\n"),
                color: state.read().theme.subtitle_color,
            )
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
    let accent_color = state.read().theme.accent_color;
    let subtitle_color = state.read().theme.subtitle_color;
    let columns_padding = state.read().theme.padding_wide();

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
