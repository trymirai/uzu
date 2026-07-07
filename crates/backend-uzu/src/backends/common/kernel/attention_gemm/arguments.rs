use crate::backends::common::gpu_types::{attention::AttnParams, ring::RingParams};

pub struct AttentionGemmArgs<Q, K, V, O, T, S> {
    pub q: Q,
    pub k: K,
    pub v: V,
    pub o: O,
    pub params: AttnParams,
    pub ring_params: Option<RingParams>,
    pub trie: Option<T>,
    pub sliding_window_size: Option<u32>,
    pub sinks: Option<S>,
    pub num_heads: u32,
    pub suffix_length: u32,
}
