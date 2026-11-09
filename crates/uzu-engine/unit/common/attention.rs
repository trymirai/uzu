use half::bf16;

use crate::backends::common::{
    Backend, Kernels,
    kernel::{AttentionKernel, AttentionKernelConfig, kv_layout::KVLayout},
};

pub fn kv_layout<B: Backend>(
    context: &B::Context,
    head_dim: u32,
    num_kv_heads: u32,
) -> KVLayout {
    let mut config =
        AttentionKernelConfig::new::<bf16>(head_dim as usize, num_kv_heads as usize, num_kv_heads as usize, true);
    config.scale = None;
    <B::Kernels as Kernels>::AttentionKernel::new(context, config).unwrap().kv_layout()
}
