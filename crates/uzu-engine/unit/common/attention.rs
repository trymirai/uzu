use crate::{
    backends::common::{
        Backend, Kernels,
        kernel::{AttentionKernel, AttentionKernelConfig, kv_layout::KVLayout},
    },
    data_type::DataType,
};

pub(crate) fn kv_layout<B: Backend>(
    context: &B::Context,
    head_dim: u32,
    num_kv_heads: u32,
) -> KVLayout {
    <B::Kernels as Kernels>::AttentionKernel::new(
        context,
        AttentionKernelConfig {
            head_dim,
            num_groups: num_kv_heads,
            num_q_heads: num_kv_heads,
            has_sinks: false,
            gate_row_stride: None,
            is_kv_cache_ring: false,
            is_causal: true,
            sliding_window_size: None,
            scale: None,
            data_type: DataType::BF16,
        },
    )
    .unwrap()
    .kv_layout()
}
