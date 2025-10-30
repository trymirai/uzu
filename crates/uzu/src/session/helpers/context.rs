use crate::{
    backends::metal::forward_pass::cache_layers::CacheLayers,
    session::config::DecodingConfig,
};

pub struct Context {
    pub tokens: Vec<u64>,
    pub cache_layers: CacheLayers,
    pub decoding_config: DecodingConfig,
}

impl Context {
    pub fn new(
        tokens: Vec<u64>,
        cache_layers: CacheLayers,
        decoding_config: DecodingConfig,
    ) -> Self {
        Self {
            tokens,
            cache_layers,
            decoding_config,
        }
    }
}
