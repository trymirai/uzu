use crate::data_type::DataType;

#[derive(Clone, Copy)]
pub struct AttentionKernelConfig {
    pub head_dim: u32,
    pub num_groups: u32,
    pub num_q_heads: u32,
    pub has_sinks: bool = false,
    pub gate_row_stride: Option<u32> = None,
    pub is_kv_cache_ring: bool = false,
    pub is_causal: bool,
    pub sliding_window_size: Option<u32> = None,
    pub scale: Option<f32> = None,
    pub data_type: DataType,
}
