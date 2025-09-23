use tokenizers::Tokenizer;

use crate::generator::config::{
    ContextLength, GeneratorConfig, GeneratorConfigProvider, SamplingSeed,
    SpeculatorConfig,
};

#[derive(Clone)]
pub struct SessionConfig {
    pub prefill_step_size: usize,
    pub context_length: ContextLength,
    pub speculator_config: SpeculatorConfig,
    pub sampling_seed: SamplingSeed,
    pub allow_pre_encode: bool,
}

impl SessionConfig {
    pub fn new(
        prefill_step_size: usize,
        context_length: ContextLength,
        speculator_config: SpeculatorConfig,
        sampling_seed: SamplingSeed,
        allow_pre_encode: bool,
    ) -> Self {
        Self {
            prefill_step_size,
            context_length,
            speculator_config,
            sampling_seed,
            allow_pre_encode,
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
            ContextLength::default(),
            SpeculatorConfig::default(),
            SamplingSeed::default(),
            true,
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
