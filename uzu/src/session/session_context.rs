use crate::{
    backends::metal::forward_pass::kv_cache::KVCache, generator::config::GeneratorConfig,
};

pub struct SessionContext {
    pub tokens: Vec<u64>,
    pub kv_cache: KVCache,
    pub config: GeneratorConfig,
}

impl SessionContext {
    pub fn new(tokens: Vec<u64>, kv_cache: KVCache, config: GeneratorConfig) -> Self {
        Self {
            tokens,
            kv_cache,
            config,
        }
    }
} 