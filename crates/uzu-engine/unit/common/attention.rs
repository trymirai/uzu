use crate::{
    backends::common::{
        Backend, Kernels,
        kernel::{AttentionKernel, AttentionKernelConfig, kv_layout::KVLayout},
    },
    data_type::DataType,
};

pub fn kv_layout<B: Backend>(
    context: &B::Context,
    head_dim: u32,
    num_kv_heads: u32,
) -> KVLayout {
    let config = AttentionKernelConfig {
        head_dim,
        num_groups: num_kv_heads,
        num_q_heads: num_kv_heads,
        is_causal: true,
        data_type: DataType::BF16,
        ..
    };
    <B::Kernels as Kernels>::AttentionKernel::new(context, config).unwrap().kv_layout()
}
