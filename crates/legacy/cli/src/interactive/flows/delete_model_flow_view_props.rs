use iocraft::prelude::*;

use crate::interactive::flows::FlowEvent;

#[derive(Default, Props)]
pub struct DeleteModelFlowViewProps {
    pub on_event: Handler<FlowEvent>,
}
