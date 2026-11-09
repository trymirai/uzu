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

#[cfg(test)]
impl AttentionKernelConfig {
    pub fn new<T: crate::data_type::DataTypeRepr>(
        head_dim: usize,
        num_q_heads: usize,
        num_groups: usize,
        is_causal: bool,
    ) -> Self {
        Self {
            head_dim: head_dim as u32,
            num_groups: num_groups as u32,
            num_q_heads: num_q_heads as u32,
            has_sinks: false,
            gate_row_stride: None,
            is_kv_cache_ring: false,
            is_causal,
            sliding_window_size: None,
            scale: Some(1.0 / (head_dim as f32).sqrt()),
            data_type: T::data_type(),
        }
    }
}
