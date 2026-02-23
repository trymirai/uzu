use crate::{backends::common::Backend, forward_pass::cache_layers::CacheLayers, session::config::DecodingConfig};

pub struct Context<B: Backend> {
    pub tokens: Vec<u64>,
    pub cache_layers: CacheLayers<B>,
    pub decoding_config: DecodingConfig,
}

impl<B: Backend> Context<B> {
    pub fn new(
        tokens: Vec<u64>,
        cache_layers: CacheLayers<B>,
        decoding_config: DecodingConfig,
    ) -> Self {
        Self {
            tokens,
            cache_layers,
            decoding_config,
        }
    }
}
