use crate::data_type::DataType;

#[derive(Clone, Copy)]
pub struct AttentionKernelConfig {
    pub head_dim: u32,
    pub num_groups: u32,
    pub num_q_heads: u32,
    pub has_sinks: bool,
    pub gate_row_stride: Option<u32>,
    pub is_kv_cache_ring: bool,
    pub is_causal: bool,
    pub sliding_window_size: Option<u32>,
    pub scale: Option<f32>,
    pub data_type: DataType,
}

impl AttentionKernelConfig {
    pub fn new(
        head_dim: u32,
        num_groups: u32,
        num_q_heads: u32,
        has_sinks: bool,
        gate_row_stride: Option<u32>,
        is_kv_cache_ring: bool,
        is_causal: bool,
        sliding_window_size: Option<u32>,
        scale: Option<f32>,
        data_type: DataType,
    ) -> Self {
        Self {
            head_dim,
            num_groups,
            num_q_heads,
            has_sinks,
            gate_row_stride,
            is_kv_cache_ring,
            is_causal,
            sliding_window_size,
            scale,
            data_type,
        }
    }
}
