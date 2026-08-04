use iocraft::prelude::*;

use crate::cli::{
    components::{ApplicationState, Gradient},
    helpers::ColorRgb,
};

#[derive(Default, Props)]
pub struct ProgressBarProps {
    pub progress: f32,
}

#[component]
pub fn ProgressBar(
    props: &ProgressBarProps,
    hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let state = hooks.use_context::<State<ApplicationState>>();
    let theme = state.read().theme.clone();
    let from_color = theme.accent_color.darker(0.5);
    let to_color = theme.accent_color;
    let track_color = theme.overlay_color();

    element! {
        View(
            flex_grow: 1.0f32,
            height: 1u32,
            background_color: Some(track_color),
        ) {
            Gradient(
                from_color: Some(from_color),
                to_color: Some(to_color),
                fill_factor: Some(props.progress),
                width: 100pct,
                height: 1u32,
            )
        }
    }
}
