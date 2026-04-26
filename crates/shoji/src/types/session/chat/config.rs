use serde::{Deserialize, Serialize};

use crate::types::{
    basic::{ContextLength, SamplingSeed},
    session::chat::ChatSpeculationPreset,
};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatConfig {
    pub context_length: ContextLength,
    pub sampling_seed: SamplingSeed,
    pub speculation_preset: Option<ChatSpeculationPreset>,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            context_length: ContextLength::default(),
            sampling_seed: SamplingSeed::default(),
            speculation_preset: None,
        }
    }
}

#[bindings::export(Implementation)]
impl ChatConfig {
    #[bindings::export(Method(Factory))]
    pub fn create() -> Self {
        Self::default()
    }
}

#[bindings::export(Implementation)]
impl ChatConfig {
    #[bindings::export(Method)]
    pub fn with_context_length(
        &self,
        context_length: ContextLength,
    ) -> Self {
        Self {
            context_length,
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_sampling_seed(
        &self,
        sampling_seed: SamplingSeed,
    ) -> Self {
        Self {
            sampling_seed,
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_speculation_preset(
        &self,
        speculation_preset: Option<ChatSpeculationPreset>,
    ) -> Self {
        Self {
            speculation_preset,
            ..self.clone()
        }
    }
}
