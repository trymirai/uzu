use crate::{backends::Backend, generator::config::GeneratorConfig};

pub struct SessionContext<B: Backend> {
    pub tokens: Vec<u64>,
    pub backend_state: B::State,
    pub config: GeneratorConfig, // TODO: shall we do anything about it?
}

impl<B: Backend> SessionContext<B> {
    pub fn new(
        tokens: Vec<u64>,
        backend_state: B::State,
        config: GeneratorConfig,
    ) -> Self {
        Self {
            tokens,
            backend_state,
            config,
        }
    }
}
