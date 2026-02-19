use std::sync::Arc;

use super::{AudioCodecRuntime, AudioTokenSpace, InputTokenAdapter, OutputTokenAdapter, TokenAdapters};

#[derive(Clone)]
pub struct AudioIntegration {
    token_space: AudioTokenSpace,
    codec_runtime: Option<Arc<dyn AudioCodecRuntime>>,
    adapters: TokenAdapters,
}

impl AudioIntegration {
    pub fn new(
        token_space: AudioTokenSpace,
        codec_runtime: Arc<dyn AudioCodecRuntime>,
        input_adapter: Arc<dyn InputTokenAdapter>,
        output_adapter: Arc<dyn OutputTokenAdapter>,
    ) -> Self {
        Self {
            token_space,
            codec_runtime: Some(codec_runtime),
            adapters: TokenAdapters::new(input_adapter, output_adapter),
        }
    }

    pub fn without_runtime(
        token_space: AudioTokenSpace,
        input_adapter: Arc<dyn InputTokenAdapter>,
        output_adapter: Arc<dyn OutputTokenAdapter>,
    ) -> Self {
        Self {
            token_space,
            codec_runtime: None,
            adapters: TokenAdapters::new(input_adapter, output_adapter),
        }
    }

    pub fn token_space(&self) -> AudioTokenSpace {
        self.token_space
    }

    pub fn codec_runtime(&self) -> Option<Arc<dyn AudioCodecRuntime>> {
        self.codec_runtime.clone()
    }

    pub fn adapters(&self) -> &TokenAdapters {
        &self.adapters
    }
}
