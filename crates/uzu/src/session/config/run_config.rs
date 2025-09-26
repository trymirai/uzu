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

    pub fn tokens_limit(
        mut self,
        tokens_limit: u64,
    ) -> Self {
        self.tokens_limit = tokens_limit;
        self
    }

    pub fn enable_thinking(
        mut self,
        enable_thinking: bool,
    ) -> Self {
        self.enable_thinking = enable_thinking;
        self
    }

    pub fn sampling_policy(
        mut self,
        sampling_policy: SamplingPolicy,
    ) -> Self {
        self.sampling_policy = sampling_policy;
        self
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self::new(1024, true, SamplingPolicy::Default)
    }
}
