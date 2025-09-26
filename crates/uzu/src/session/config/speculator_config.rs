use std::sync::Arc;

use crate::speculators::{
    empty_speculator::EmptySpeculator, speculator::Speculator,
};

#[derive(Clone)]
pub struct SpeculatorConfig {
    pub number_of_speculated_tokens: usize,
    pub speculator: Arc<dyn Speculator>,
}

impl SpeculatorConfig {
    pub fn new(
        number_of_speculated_tokens: usize,
        speculator: Arc<dyn Speculator>,
    ) -> Self {
        Self {
            number_of_speculated_tokens,
            speculator,
        }
    }
}

impl Default for SpeculatorConfig {
    fn default() -> Self {
        Self {
            number_of_speculated_tokens: 0,
            speculator: Arc::new(EmptySpeculator {}),
        }
    }
}
