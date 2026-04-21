use serde::{Deserialize, Serialize};

use crate::types::basic::SamplingMethod;

#[bindings::export(Enum, name = "SamplingPolicy")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SamplingPolicy {
    Default {},
    Custom {
        method: SamplingMethod,
    },
}

impl Default for SamplingPolicy {
    fn default() -> Self {
        SamplingPolicy::Default {}
    }
}
