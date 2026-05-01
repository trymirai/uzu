use iocraft::prelude::*;

use crate::cli::{
    components::ApplicationState,
    helpers::{SYMBOL_COMMAND, SYMBOL_INPUT, SYMBOL_INPUT_RESULT},
};

#[derive(Clone)]
pub enum HistoryCellType {
    Command {
        name: String,
    },
    CommandResult {
        result: String,
    },
}

#[derive(Default, Props)]
pub struct HistoryCellProps {
    pub r#type: Option<HistoryCellType>,
}

#[component]
pub fn HistoryCell(
    props: &HistoryCellProps,
    hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let state = hooks.use_context::<State<ApplicationState>>();
    let theme = state.read().theme.clone();

    let view: AnyElement<'static> = match props.r#type.clone() {
        Some(HistoryCellType::Command {
            name,
        }) => element! {
            View(flex_direction: FlexDirection::Row, column_gap: theme.padding()) {
                Text(content: SYMBOL_INPUT, color: theme.accent_color)
                Text(content: format!("{}{}", SYMBOL_COMMAND, name), weight: Weight::Bold, color: theme.accent_color)
            }
        }
        .into(),
        Some(HistoryCellType::CommandResult {
            result,
        }) => element! {
            View(flex_direction: FlexDirection::Row, column_gap: theme.padding(), padding_left: 2 * theme.padding()) {
                Text(content: SYMBOL_INPUT_RESULT, color: theme.subtitle_color)
                Text(content: result)
            }
        }
        .into(),
        None => element! { View }.into(),
    };
    view
}
