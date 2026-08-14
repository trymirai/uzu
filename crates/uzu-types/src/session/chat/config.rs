use crate::common::{ContextLength, SamplingSeed};

#[derive(Clone, Default)]
pub struct ChatSessionConfig {
    pub context_length: ContextLength,
    pub sampling_seed: SamplingSeed,
}

impl ChatSessionConfig {
    pub fn create() -> Self {
        Self::default()
    }
}

impl ChatSessionConfig {
    pub fn with_context_length(
        &self,
        context_length: ContextLength,
    ) -> Self {
        Self {
            context_length,
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
}
