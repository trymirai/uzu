#[derive(Debug, Clone)]
pub struct TraceDebugLayerSnapshot {
    pub layer_index: usize,
    pub model_dim: usize,
    pub active_row_count: usize,
    pub sparse_value_single_decode_has_kv_cache: bool,
    pub sparse_value_single_decode_has_sparse_value: bool,
    pub sparse_value_single_decode_suffix_length: usize,
    pub sparse_value_single_decode_projection_step: usize,
    pub sparse_value_single_decode_is_trie: bool,
    pub sparse_value_single_decode_is_kv_cache_ring: bool,
    pub attempted_sparse_value_single_decode: bool,
    pub used_sparse_value_single_decode: bool,
    pub pre_attention_norm: Vec<f32>,
    pub attention: Vec<f32>,
    pub sparse_expected_attention: Option<Vec<f32>>,
    pub outputs: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TraceDebugSnapshot {
    pub layers: Vec<TraceDebugLayerSnapshot>,
}
