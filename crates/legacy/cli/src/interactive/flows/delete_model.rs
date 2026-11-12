use iocraft::prelude::*;

use crate::interactive::flows::{DeleteModelFlowView, Flow, FlowEvent};

pub struct DeleteModelFlow;

impl Flow for DeleteModelFlow {
    fn render(
        &self,
        on_event: Handler<FlowEvent>,
    ) -> AnyElement<'static> {
        element! { DeleteModelFlowView(on_event: on_event) }.into()
    }
}
