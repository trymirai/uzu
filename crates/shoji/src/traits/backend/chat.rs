use serde::{Deserialize, Serialize};

use crate::types::session::chat::SamplingPolicy;

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamConfig {
    pub token_limit: Option<u32>,
    pub sampling_policy: SamplingPolicy,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            token_limit: None,
            sampling_policy: SamplingPolicy::default(),
        }
    }
}
