use serde::{Deserialize, Serialize};

use crate::types::{
    basic::{SamplingMethod, SamplingPolicy},
    session::chat::Grammar,
};

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatStreamConfig {
    pub token_limit: Option<u32>,
    pub sampling_policy: SamplingPolicy,
    pub grammar: Option<Grammar>,
}

impl Default for ChatStreamConfig {
    fn default() -> Self {
        Self {
            token_limit: None,
            sampling_policy: SamplingPolicy::default(),
            grammar: None,
        }
    }
}

impl ChatStreamConfig {
    pub fn with_token_limit(
        self,
        token_limit: Option<u32>,
    ) -> Self {
        Self {
            token_limit,
            ..self
        }
    }

    pub fn with_sampling_policy(
        self,
        sampling_policy: SamplingPolicy,
    ) -> Self {
        Self {
            sampling_policy,
            ..self
        }
    }

    pub fn with_sampling_method(
        self,
        sampling_method: SamplingMethod,
    ) -> Self {
        Self {
            sampling_policy: SamplingPolicy::Custom {
                method: sampling_method,
            },
            ..self
        }
    }

    pub fn with_grammar(
        self,
        grammar: Option<Grammar>,
    ) -> Self {
        Self {
            grammar,
            ..self
        }
    }
}
