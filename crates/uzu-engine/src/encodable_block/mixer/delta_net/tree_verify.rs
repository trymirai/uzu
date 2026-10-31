use crate::{
    backends::common::{Backend, Kernels},
    data_type::DataType,
    encodable_block::mixer::delta_net::accepted_path::AcceptedPath,
};

#[derive(Clone, Copy)]
pub struct TreeVerifyNewArguments {
    pub data_type: DataType,
    pub num_k_heads: u32,
    pub num_v_heads: u32,
    pub head_k_dim: u32,
    pub head_v_dim: u32,
}

pub struct TreeVerifyEncodeArguments<'a, B: Backend> {
    pub q: &'a B::ScratchBuffer,
    pub k: &'a B::ScratchBuffer,
    pub v: &'a B::ScratchBuffer,
    pub trie: &'a B::ConstantBuffer,
    pub log_decay: &'a B::ScratchBuffer,
    pub beta: &'a B::ScratchBuffer,
    /// The state, which first advances in place along `advance`, if any.
    pub h0: &'a mut B::GlobalGpuBuffer,
    pub advance: Option<(&'a AcceptedPath<B>, &'a <B::Kernels as Kernels>::StateAdvanceKernel)>,
    pub in_proj: &'a B::ScratchBuffer,
    pub norm_weight: &'a B::GlobalGpuBuffer,
    pub conv_dim: u32,
    pub total_proj_dim: u32,
    pub norm_epsilon: f32,
    pub tree_size: u32,
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/delta_net_tree_verify_bench.rs"]
mod tests;
