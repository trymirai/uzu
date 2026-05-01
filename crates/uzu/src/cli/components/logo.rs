use iocraft::prelude::*;

use crate::cli::components::ApplicationState;

#[component]
pub fn Logo(hooks: Hooks) -> impl Into<AnyElement<'static>> {
    let state = hooks.use_context::<State<ApplicationState>>();

    element! {
        View(
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::FlexStart,
            width: 100pct,
            column_gap: state.read().theme.padding_wide(),
        ) {
            Text(
                content: state.read().theme.logo(),
                color: state.read().theme.accent_color
            )
            Text(
                content: state.read().theme.about(),
                color: state.read().theme.subtitle_color
            )
        }
    }
}
