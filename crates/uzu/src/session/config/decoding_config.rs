use crate::session::{
    config::SpeculatorConfig,
    parameter::{
        AsyncBatchSize, ContextLength, ContextMode, PrefillStepSize,
        SamplingSeed,
    },
};

#[derive(Clone)]
pub struct DecodingConfig {
    pub context_mode: ContextMode,
    pub context_length: ContextLength,
    pub prefill_step_size: PrefillStepSize,
    pub speculator_config: SpeculatorConfig,
    pub sampling_seed: SamplingSeed,
    pub async_batch_size: AsyncBatchSize,
    pub allow_pre_encode: bool,
}

impl DecodingConfig {
    pub fn new(
        context_mode: ContextMode,
        context_length: ContextLength,
        prefill_step_size: PrefillStepSize,
        speculator_config: SpeculatorConfig,
        sampling_seed: SamplingSeed,
        async_batch_size: AsyncBatchSize,
        allow_pre_encode: bool,
    ) -> Self {
        Self {
            context_mode,
            context_length,
            prefill_step_size,
            speculator_config,
            sampling_seed,
            async_batch_size,
            allow_pre_encode,
        }
    }

    pub fn generate_suffix_length(&self) -> usize {
        self.speculator_config.number_of_speculated_tokens + 1
    }
}

impl Default for DecodingConfig {
    fn default() -> Self {
        Self {
            context_mode: ContextMode::default(),
            context_length: ContextLength::default(),
            prefill_step_size: PrefillStepSize::default(),
            speculator_config: SpeculatorConfig::default(),
            sampling_seed: SamplingSeed::default(),
            async_batch_size: AsyncBatchSize::default(),
            allow_pre_encode: true,
        }
    }
}

impl DecodingConfig {
    pub fn with_context_mode(
        &self,
        context_mode: ContextMode,
    ) -> Self {
        Self {
            context_mode,
            ..self.clone()
        }
    }

    pub fn with_context_length(
        &self,
        context_length: ContextLength,
    ) -> Self {
        Self {
            context_length,
            ..self.clone()
        }
    }

    pub fn with_prefill_step_size(
        &self,
        prefill_step_size: PrefillStepSize,
    ) -> Self {
        Self {
            prefill_step_size,
            ..self.clone()
        }
    }

    pub fn with_speculator_config(
        &self,
        speculator_config: SpeculatorConfig,
    ) -> Self {
        Self {
            speculator_config,
            ..self.clone()
        }
    }

    pub fn with_sampling_seed(
        &self,
        sampling_seed: SamplingSeed,
    ) -> Self {
        Self {
            sampling_seed,
            ..self.clone()
        }
    }

    pub fn with_async_batch_size(
        &self,
        async_batch_size: AsyncBatchSize,
    ) -> Self {
        Self {
            async_batch_size,
            ..self.clone()
        }
    }

    pub fn with_allow_pre_encode(
        &self,
        allow_pre_encode: bool,
    ) -> Self {
        Self {
            allow_pre_encode,
            ..self.clone()
        }
    }
}
