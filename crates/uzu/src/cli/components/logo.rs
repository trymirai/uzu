use iocraft::prelude::*;

use crate::cli::components::Theme;

#[component]
pub fn Logo(hooks: Hooks) -> impl Into<AnyElement<'static>> {
    let theme = hooks.use_context::<Theme>();

    element! {
        View(
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::FlexStart,
            width: 100pct,
            column_gap: theme.padding_wide(),
        ) {
            Text(
                content: theme.logo(),
                color: theme.accent_color
            )
            Text(
                content: theme.about(),
                color: theme.subtitle_color
            )
        }
    }
}
