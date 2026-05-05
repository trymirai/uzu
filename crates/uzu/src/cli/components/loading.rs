use iocraft::prelude::*;

use crate::cli::components::ApplicationState;

const DEFAULT_LABEL: &str = "Loading...";

#[derive(Default, Props)]
pub struct LoadingProps<'a> {
    pub loaded: bool,
    pub label: Option<String>,
    pub children: Vec<AnyElement<'a>>,
}

#[component]
pub fn Loading<'a>(
    props: &mut LoadingProps<'a>,
    hooks: Hooks,
) -> impl Into<AnyElement<'a>> {
    let label = if let Some(label) = props.label.clone() {
        label
    } else {
        DEFAULT_LABEL.to_string()
    };
    let children = std::mem::take(&mut props.children);

    let state = hooks.use_context::<State<ApplicationState>>();

    let view: AnyElement<'a> = if props.loaded {
        element! {
            View(flex_direction: FlexDirection::Column, width: 100pct) {
                #(children.into_iter())
            }
        }
        .into()
    } else {
        element! {
            Text(content: label, color: state.read().theme.subtitle_color)
        }
        .into()
    };
    view
}
