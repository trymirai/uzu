use crate::{
    backends::common::{Allocation, Backend},
    data_type::DataType,
};

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
    pub q: &'a Allocation<B>,
    pub k: &'a Allocation<B>,
    pub v: &'a Allocation<B>,
    pub trie: &'a Allocation<B>,
    pub log_decay: &'a Allocation<B>,
    pub beta: &'a Allocation<B>,
    pub h0: &'a Allocation<B>,
    pub tree_size: u32,
}

#[cfg(test)]
#[path = "../../../../tests/unit/encodable_block/delta_net_tree_verify_bench.rs"]
mod tests;
