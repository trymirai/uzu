use crate::{
    backends::common::{Allocation, Backend},
    data_type::DataType,
};

#[derive(Clone, Copy)]
pub struct TreeVerifyNewArguments {
    pub data_type: DataType,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
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
    pub tree_size: usize,
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/delta_net_tree_verify_bench.rs"]
mod tests;
