use crate::session::{
    config::SpeculatorConfig,
    parameter::{ContextLength, PrefillStepSize, SamplingSeed},
    types::Input,
};

#[derive(Clone)]
pub struct DecodingConfig {
    pub prefill_step_size: PrefillStepSize,
    pub context_length: ContextLength,
    pub speculator_config: SpeculatorConfig,
    pub sampling_seed: SamplingSeed,
    pub allow_pre_encode: bool,
    pub context_mode: ContextMode,
}

#[derive(Clone, Debug)]
pub enum ContextMode {
    None,
    Static {
        input: Input,
    },
    Dynamic,
}

impl DecodingConfig {
    pub fn new(
        prefill_step_size: PrefillStepSize,
        context_length: ContextLength,
        speculator_config: SpeculatorConfig,
        sampling_seed: SamplingSeed,
        allow_pre_encode: bool,
        context_mode: ContextMode,
    ) -> Self {
        Self {
            prefill_step_size,
            context_length,
            speculator_config,
            sampling_seed,
            allow_pre_encode,
            context_mode,
        }
    }

    pub fn generate_suffix_length(&self) -> usize {
        self.speculator_config.number_of_speculated_tokens + 1
    }
}

impl Default for DecodingConfig {
    fn default() -> Self {
        Self {
            prefill_step_size: PrefillStepSize::default(),
            context_length: ContextLength::default(),
            speculator_config: SpeculatorConfig::default(),
            sampling_seed: SamplingSeed::default(),
            allow_pre_encode: true,
            context_mode: ContextMode::None,
        }
    }
}
