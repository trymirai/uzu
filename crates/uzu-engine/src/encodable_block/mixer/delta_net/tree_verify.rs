use crate::{backends::common::Backend, data_type::DataType};

#[derive(Clone, Copy)]
pub struct TreeVerifyNewArguments {
    pub data_type: DataType,
    pub num_k_heads: u32,
    pub num_v_heads: u32,
    pub head_k_dim: u32,
    pub head_v_dim: u32,
}

#[derive(Clone, Copy)]
pub struct TreeVerifyEncodeArguments<'a, B: Backend> {
    pub q: &'a B::ScratchBuffer,
    pub k: &'a B::ScratchBuffer,
    pub v: &'a B::ScratchBuffer,
    pub trie: &'a B::ConstantBuffer,
    pub log_decay: &'a B::ScratchBuffer,
    pub beta: &'a B::ScratchBuffer,
    pub h0: &'a B::GlobalBuffer,
    pub tree_size: u32,
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/delta_net_tree_verify_bench.rs"]
mod tests;
