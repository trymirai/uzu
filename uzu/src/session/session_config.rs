use tokenizers::Tokenizer;

use super::sampling_config::SamplingConfig;
use crate::generator::config::{
    ContextLength, GeneratorConfig, GeneratorConfigProvider, SamplingSeed,
    SpeculatorConfig,
};

#[derive(Clone)]
pub struct SessionConfig {
    pub prefill_step_size: usize,
    pub prefix_length_step: Option<usize>,
    pub speculator_config: SpeculatorConfig,
    pub allow_pre_encode: bool,
    pub sampling_seed: SamplingSeed,
    pub context_length: ContextLength,
}

impl SessionConfig {
    pub fn new(
        prefill_step_size: usize,
        prefix_length_step: Option<usize>,
        speculator_config: SpeculatorConfig,
        allow_pre_encode: bool,
        sampling_seed: SamplingSeed,
        context_length: ContextLength,
    ) -> Self {
        Self {
            prefill_step_size,
            prefix_length_step,
            speculator_config,
            allow_pre_encode,
            sampling_seed,
            context_length,
        }
    }

    pub fn to_generator_config(&self) -> GeneratorConfig {
        GeneratorConfig::new(
            self.prefill_step_size,
            self.prefix_length_step,
            self.speculator_config.clone(),
            self.allow_pre_encode,
            self.sampling_seed,
            self.context_length,
        )
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self::new(
            8,
            None,
            SpeculatorConfig::default(),
            true,
            SamplingSeed::default(),
            ContextLength::default(),
        )
    }
}

impl GeneratorConfigProvider for SessionConfig {
    fn generator_config(
        &self,
        _tokenizer: &Tokenizer,
    ) -> GeneratorConfig {
        self.to_generator_config()
    }
}

#[derive(Debug)]
pub struct SessionRunConfig {
    pub tokens_limit: u64,
    pub sampling_method: SamplingConfig,
}

impl SessionRunConfig {
    pub fn new(tokens_limit: u64) -> Self {
        Self {
            tokens_limit: tokens_limit,
            sampling_method: SamplingConfig::default(),
        }
    }

    pub fn new_with_sampling(
        tokens_limit: u64,
        sampling_method: SamplingConfig,
    ) -> Self {
        Self {
            tokens_limit: tokens_limit,
            sampling_method,
        }
    }
}
