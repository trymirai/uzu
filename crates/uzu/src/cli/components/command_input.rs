use iocraft::prelude::*;
use unicode_width::UnicodeWidthStr;

use crate::cli::{
    components::{ApplicationState, TextInput},
    helpers::SYMBOL_INPUT,
};

const SAFE_PADDING: u16 = 1;

#[derive(Default, Props)]
pub struct CommandInputProps {
    pub on_submit: HandlerMut<'static, String>,
}

#[component]
pub fn CommandInput(
    props: &mut CommandInputProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let on_submit = props.on_submit.take();

    let state = hooks.use_context::<State<ApplicationState>>();
    let (width, _) = hooks.use_terminal_size();

    element! {
        View(flex_direction: FlexDirection::Column) {
            View(
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::FlexStart,
                width: 100pct,
                column_gap: state.read().theme.padding(),
                border_color: state.read().theme.subtitle_color,
                border_style: BorderStyle::Single,
                border_edges: Some(Edges::Top | Edges::Bottom),
            ) {
                Text(content: SYMBOL_INPUT, color: state.read().theme.accent_color)
                TextInput(
                    maximal_width: width.saturating_sub(UnicodeWidthStr::width(SYMBOL_INPUT) as u16).saturating_sub(state.read().theme.padding()).saturating_sub(SAFE_PADDING),
                    has_focus: true,
                    on_submit: on_submit,
                )
            }
            Text(content: state.read().theme.default_hint(), color: state.read().theme.subtitle_color)
        }
    }
}
