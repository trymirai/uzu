use iocraft::prelude::*;
use unicode_width::UnicodeWidthStr;

use crate::cli::components::{TextInput, Theme};

const SYMBOL_ARROW: &str = "❯";
const SAFE_PADDING: u16 = 1;

#[component]
pub fn CommandInput(mut hooks: Hooks) -> impl Into<AnyElement<'static>> {
    let theme = hooks.use_context::<Theme>();
    let (width, _) = hooks.use_terminal_size();

    element! {
        View(
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::FlexStart,
            width: 100pct,
            column_gap: theme.padding(),
            border_color: theme.subtitle_color,
            border_style: BorderStyle::Single,
            border_edges: Some(Edges::Top | Edges::Bottom),
        ) {
            Text(content: SYMBOL_ARROW, color: theme.accent_color)
            TextInput(
                maximal_width: width.saturating_sub(UnicodeWidthStr::width(SYMBOL_ARROW) as u16).saturating_sub(theme.padding()).saturating_sub(SAFE_PADDING),
                has_focus: true,
                on_submit: move |_new: String| {},
            )
        }
    }
}
