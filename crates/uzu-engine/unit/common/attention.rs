use half::bf16;

use crate::{
    backends::common::{
        Backend, Kernels,
        kernel::{AttentionKernel, AttentionKernelConfig, kv_layout::KVLayout},
    },
    data_type::DataTypeRepr,
};

pub fn default_attention_config<T: DataTypeRepr>(
    head_dim: usize,
    num_q_heads: usize,
    num_groups: usize,
    is_causal: bool,
) -> AttentionKernelConfig {
    AttentionKernelConfig {
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

pub(crate) fn kv_layout<B: Backend>(
    context: &B::Context,
    head_dim: u32,
    num_kv_heads: u32,
) -> KVLayout {
    let config =
        default_attention_config::<bf16>(head_dim as usize, num_kv_heads as usize, num_kv_heads as usize, true);
    <B::Kernels as Kernels>::AttentionKernel::new(context, config).unwrap().kv_layout()
}
