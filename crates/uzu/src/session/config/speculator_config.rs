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
        let max_number_of_speculated_tokens: usize = 64;
        let effective_number_of_speculated_tokens = std::cmp::min(
            number_of_speculated_tokens,
            max_number_of_speculated_tokens,
        );
        Self {
            number_of_speculated_tokens: effective_number_of_speculated_tokens,
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
