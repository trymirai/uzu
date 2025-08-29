use crate::{
    backends::metal::MTLContext,
    generator::{KVCache, config::GeneratorConfig},
};

pub struct SessionContext {
    pub tokens: Vec<u64>,
    pub kv_cache: KVCache<MTLContext>,
    pub config: GeneratorConfig,
}

impl SessionContext {
    pub fn new(
        tokens: Vec<u64>,
        kv_cache: KVCache<MTLContext>,
        config: GeneratorConfig,
    ) -> Self {
        Self {
            tokens,
            kv_cache,
            config,
        }
    }
}
