use crate::session::parameter::SamplingPolicy;

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub tokens_limit: u64,
    pub enable_thinking: bool,
    pub sampling_policy: SamplingPolicy,
}

impl RunConfig {
    pub fn new(
        tokens_limit: u64,
        enable_thinking: bool,
        sampling_policy: SamplingPolicy,
    ) -> Self {
        Self {
            tokens_limit,
            enable_thinking,
            sampling_policy,
        }
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self::new(1024, true, SamplingPolicy::Default)
    }
}
