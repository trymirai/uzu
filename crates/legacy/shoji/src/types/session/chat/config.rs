use serde::{Deserialize, Serialize};

use super::SpeculationMode;
use crate::types::basic::{ContextLength, SamplingSeed};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatConfig {
    pub context_length: ContextLength,
    pub sampling_seed: SamplingSeed,
    #[serde(default)]
    pub speculation: SpeculationMode,
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
    pub fn with_speculation(
        &self,
        speculation: SpeculationMode,
    ) -> Self {
        Self {
            speculation,
            ..self.clone()
        }
    }
}
