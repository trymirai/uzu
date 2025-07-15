use crate::backends::metal::sampling_config::SamplingConfig;

#[derive(Debug)]
pub struct SessionRunConfig {
    pub tokens_limit: u64,
    pub sampling_config: Option<SamplingConfig>,
}

impl SessionRunConfig {
    pub fn new(tokens_limit: u64) -> Self {
        Self {
            tokens_limit: tokens_limit,
            sampling_config: None,
        }
    }

    pub fn new_with_sampling_config(
        tokens_limit: u64,
        sampling_config: Option<SamplingConfig>,
    ) -> Self {
        Self {
            tokens_limit: tokens_limit,
            sampling_config: sampling_config,
        }
    }
}
