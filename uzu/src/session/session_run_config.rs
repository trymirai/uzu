use crate::backends::metal::sampling_config::SamplingConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunMode {
    Stateless,
    WithPrefix,
}

impl Default for RunMode {
    fn default() -> Self {
        RunMode::Stateless
    }
}

#[derive(Debug)]
pub struct SessionRunConfig {
    pub tokens_limit: u64,
    pub sampling_config: Option<SamplingConfig>,
    pub run_mode: RunMode,
}

impl SessionRunConfig {
    pub fn new(tokens_limit: u64) -> Self {
        Self {
            tokens_limit: tokens_limit,
            sampling_config: None,
            run_mode: RunMode::Stateless,
        }
    }

    pub fn new_with_sampling_config(
        tokens_limit: u64,
        sampling_config: Option<SamplingConfig>,
    ) -> Self {
        Self {
            tokens_limit: tokens_limit,
            sampling_config: sampling_config,
            run_mode: RunMode::Stateless,
        }
    }

    pub fn with_run_mode(
        mut self,
        mode: RunMode,
    ) -> Self {
        self.run_mode = mode;
        self
    }
}
