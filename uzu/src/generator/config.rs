use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::speculators::{
    empty_speculator::EmptySpeculator, speculator::Speculator,
};

#[derive(Clone)]
pub struct SpeculatorConfig {
    pub number_of_speculated_tokens: usize,
    pub speculator: Arc<dyn Speculator>,
}

impl Default for SpeculatorConfig {
    fn default() -> Self {
        Self {
            number_of_speculated_tokens: 0,
            speculator: Arc::new(EmptySpeculator {}),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ContextLength {
    // 8192 is the default max prefix length
    Default,
    // Custom max prefix length
    Custom(u64),
}

impl Default for ContextLength {
    fn default() -> Self {
        ContextLength::Default
    }
}

impl ContextLength {
    pub fn get_value(&self) -> u64 {
        match self {
            ContextLength::Default => 8192,
            ContextLength::Custom(length) => *length,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SamplingSeed {
    // 42 is the default sampling seed
    Default,
    // Custom sampling seed
    Custom(u64),
}

impl Default for SamplingSeed {
    fn default() -> Self {
        SamplingSeed::Default
    }
}

impl SamplingSeed {
    pub fn get_value(&self) -> u64 {
        match self {
            SamplingSeed::Default => 42,
            SamplingSeed::Custom(seed) => *seed,
        }
    }
}

pub trait GeneratorConfigProvider {
    fn generator_config(
        &self,
        tokenizer: &Tokenizer,
    ) -> GeneratorConfig;
}

pub struct GeneratorConfig {
    pub prefill_step_size: usize,
    pub speculator_config: SpeculatorConfig,
    pub allow_pre_encode: bool,
    pub sampling_seed: u64,
    pub context_length: usize,
}

impl GeneratorConfig {
    pub fn new(
        prefill_step_size: usize,
        speculator_config: SpeculatorConfig,
        allow_pre_encode: bool,
        sampling_seed: SamplingSeed,
        context_length: ContextLength,
    ) -> Self {
        Self {
            prefill_step_size,
            speculator_config,
            allow_pre_encode,
            sampling_seed: sampling_seed.get_value(),
            context_length: context_length.get_value() as usize,
        }
    }

    pub fn generate_suffix_length(&self) -> usize {
        self.speculator_config.number_of_speculated_tokens + 1
    }
}
