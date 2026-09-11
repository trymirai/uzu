use serde::{Deserialize, Serialize};

use crate::types::basic::{Grammar, SamplingMethod, SamplingPolicy};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatReplyConfig {
    pub token_limit: Option<u32>,
    pub sampling_policy: SamplingPolicy,
    pub grammar: Option<Grammar>,
    /// Maximum number of automatic tool-call turns per reply.
    /// `None` falls back to the session default.
    pub tool_turn_limit: Option<u32>,
}

#[bindings::export(Implementation)]
impl ChatReplyConfig {
    #[bindings::export(Method(Factory))]
    pub fn create() -> Self {
        Self::default()
    }
}

#[bindings::export(Implementation)]
impl ChatReplyConfig {
    #[bindings::export(Method)]
    pub fn with_token_limit(
        &self,
        token_limit: Option<u32>,
    ) -> Self {
        Self {
            token_limit,
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_sampling_policy(
        &self,
        sampling_policy: SamplingPolicy,
    ) -> Self {
        Self {
            sampling_policy,
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_sampling_method(
        &self,
        sampling_method: SamplingMethod,
    ) -> Self {
        Self {
            sampling_policy: SamplingPolicy::Custom {
                method: sampling_method,
            },
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_tool_turn_limit(
        &self,
        tool_turn_limit: Option<u32>,
    ) -> Self {
        Self {
            tool_turn_limit,
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_grammar(
        &self,
        grammar: Option<Grammar>,
    ) -> Self {
        Self {
            grammar,
            ..self.clone()
        }
    }
}
