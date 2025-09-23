use tokenizers::Tokenizer;

use crate::generator::config::{
    ContextLength, GeneratorConfig, GeneratorConfigProvider, SamplingSeed,
    SpeculatorConfig,
};

#[derive(Clone)]
pub struct SessionConfig {
    pub prefill_step_size: usize,
    pub speculator_config: SpeculatorConfig,
    pub allow_pre_encode: bool,
    pub sampling_seed: SamplingSeed,
    pub context_length: ContextLength,
}

impl SessionConfig {
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
            sampling_seed,
            context_length,
        }
    }

    pub fn to_generator_config(&self) -> GeneratorConfig {
        GeneratorConfig::new(
            self.prefill_step_size,
            self.speculator_config.clone(),
            self.allow_pre_encode,
            self.sampling_seed,
            self.context_length,
        )
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        let prefill_step_size: usize;
        if cfg!(target_os = "ios") {
            prefill_step_size = 64;
        } else {
            prefill_step_size = 256;
        }

        Self::new(
            prefill_step_size,
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
