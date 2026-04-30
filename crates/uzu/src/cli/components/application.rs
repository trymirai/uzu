use iocraft::prelude::*;

use crate::{
    cli::components::{CommandInput, Logo, Theme},
    engine::Engine,
};

#[derive(Default, Props)]
pub struct ApplicationProps {
    pub engine: Option<Engine>,
}

#[component]
pub fn Application(
    props: &ApplicationProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let engine = props.engine.clone().expect("Application requires an engine");
    let theme = hooks.use_state(|| Theme::default());
    let (width, _) = hooks.use_terminal_size();

    element! {
        ContextProvider(value: Context::owned(engine)) {
            ContextProvider(value: Context::owned(theme.read().clone())) {
                View(
                    flex_direction: FlexDirection::Column,
                    width: width as u16,
                    row_gap: theme.read().padding()
                ) {
                    View (
                        padding_left: theme.read().padding(),
                        padding_right: theme.read().padding(),
                    ) {
                        Logo
                    }
                    CommandInput
                }
            }
        }
    }
}
