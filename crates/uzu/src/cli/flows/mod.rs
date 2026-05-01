mod exit;
mod registry;
mod theme;

pub use exit::ExitFlow;
use iocraft::prelude::*;
pub use registry::{Command, FlowRegistry};
pub use theme::ThemeFlow;

pub struct FlowEvent {
    pub result: String,
    pub next_flow: Option<Box<dyn Flow>>,
}

impl FlowEvent {
    pub fn finish(result: impl Into<String>) -> Self {
        Self {
            result: result.into(),
            next_flow: None,
        }
    }

    pub fn transition(
        result: impl Into<String>,
        next: Box<dyn Flow>,
    ) -> Self {
        Self {
            result: result.into(),
            next_flow: Some(next),
        }
    }
}

pub trait Flow: Send + Sync {
    fn render(
        &self,
        on_event: Handler<FlowEvent>,
    ) -> AnyElement<'static>;
}
