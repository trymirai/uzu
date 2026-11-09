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
    let config = AttentionKernelConfig::new(
        head_dim,
        num_kv_heads,
        num_kv_heads,
        false,
        None,
        false,
        true,
        None,
        None,
        DataType::BF16,
    );
    <B::Kernels as Kernels>::AttentionKernel::new(context, config).unwrap().kv_layout()
}
