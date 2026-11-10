use serde::{Deserialize, Serialize};

use crate::types::basic::{Grammar, SamplingMethod, SamplingPolicy, SamplingSeed};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatReplyConfig {
    pub token_limit: Option<u32>,
    pub sampling_policy: SamplingPolicy,
    pub grammar: Option<Grammar>,
    /// Maximum number of automatic tool-call turns per reply.
    /// `None` allows unlimited turns; `Some(0)` prevents automatic tool execution.
    pub tool_turn_limit: Option<u32>,
    /// Seed for this reply only. `Default` keeps the session's seed from `ChatConfig`.
    pub sampling_seed: SamplingSeed,
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
}
