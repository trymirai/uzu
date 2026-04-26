use serde::{Deserialize, Serialize};

use crate::types::basic::SamplingMethod;

#[bindings::export(Enumeration)]
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
