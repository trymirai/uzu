use crate::{
    backends::metal::forward_pass::kv_cache::KVCache,
    session::config::DecodingConfig,
};

pub struct SessionContext {
    pub tokens: Vec<u64>,
    pub kv_cache: KVCache,
    pub decoding_config: DecodingConfig,
}

impl SessionContext {
    pub fn new(
        tokens: Vec<u64>,
        kv_cache: KVCache,
        decoding_config: DecodingConfig,
    ) -> Self {
        Self {
            tokens,
            kv_cache,
            decoding_config,
        }
    }
}
