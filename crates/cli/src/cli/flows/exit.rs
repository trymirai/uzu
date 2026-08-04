use iocraft::prelude::*;

use crate::cli::{
    components::ApplicationState,
    flows::{Flow, FlowEvent},
};

pub struct ExitFlow;

impl Flow for ExitFlow {
    fn render(
        &self,
        _on_event: Handler<FlowEvent>,
    ) -> AnyElement<'static> {
        element! { ExitFlowView }.into()
    }
}

#[component]
fn ExitFlowView(hooks: Hooks) -> impl Into<AnyElement<'static>> {
    let state = *hooks.use_context::<State<ApplicationState>>();

    let mut system = hooks.use_context_mut::<SystemContext>();
    system.exit();

    element! { Text(content: format!("See you soon {}!", state.read().theme.symbol_heart), color: state.read().theme.accent_color) }
}
