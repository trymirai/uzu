use std::collections::HashMap;

pub trait Speculator: Send + Sync {
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32>;
}
