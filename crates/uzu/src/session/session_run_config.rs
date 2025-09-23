use crate::backends::metal::sampling_config::SamplingConfig;

#[derive(Debug)]
pub struct SessionRunConfig {
    pub tokens_limit: u64,
    pub enable_thinking: bool,
    pub custom_sampling_config: Option<SamplingConfig>,
}

impl SessionRunConfig {
    pub fn new(
        tokens_limit: u64,
        enable_thinking: bool,
        custom_sampling_config: Option<SamplingConfig>,
    ) -> Self {
        Self {
            tokens_limit: tokens_limit,
            enable_thinking: enable_thinking,
            custom_sampling_config: custom_sampling_config,
        }
    }
}

impl Default for SessionRunConfig {
    fn default() -> Self {
        Self::new(1024, true, None)
    }
}
