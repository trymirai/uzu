use serde::{Deserialize, Serialize};

use crate::types::{
    basic::SamplingSeed,
    session::chat::{ChatContextLength, ChatSpeculationPreset},
};

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatConfig {
    pub context_length: ChatContextLength,
    pub sampling_seed: SamplingSeed,
    pub speculation_preset: Option<ChatSpeculationPreset>,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            context_length: ChatContextLength::default(),
            sampling_seed: SamplingSeed::default(),
            speculation_preset: None,
        }
    }
}

impl ChatConfig {
    pub fn with_context_length(
        self,
        context_length: ChatContextLength,
    ) -> Self {
        Self {
            context_length,
            ..self
        }
    }
}

impl ChatConfig {
    pub fn with_sampling_seed(
        self,
        sampling_seed: SamplingSeed,
    ) -> Self {
        Self {
            sampling_seed,
            ..self
        }
    }
}

impl ChatConfig {
    pub fn with_speculation_preset(
        self,
        speculation_preset: Option<ChatSpeculationPreset>,
    ) -> Self {
        Self {
            speculation_preset,
            ..self
        }
    }
}
