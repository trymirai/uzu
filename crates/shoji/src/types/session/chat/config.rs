use serde::{Deserialize, Serialize};

use crate::types::{
    basic::SamplingSeed,
    session::chat::{ContextLength, SpeculationPreset},
};

#[bindings::export(Struct, name = "ChatConfig")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub context_length: ContextLength,
    pub sampling_seed: SamplingSeed,
    pub speculation_preset: Option<SpeculationPreset>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            context_length: ContextLength::default(),
            sampling_seed: SamplingSeed::default(),
            speculation_preset: None,
        }
    }
}

impl Config {
    pub fn with_context_length(
        self,
        context_length: ContextLength,
    ) -> Self {
        Self {
            context_length,
            ..self
        }
    }
}

impl Config {
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

impl Config {
    pub fn with_speculation_preset(
        self,
        speculation_preset: Option<SpeculationPreset>,
    ) -> Self {
        Self {
            speculation_preset,
            ..self
        }
    }
}
