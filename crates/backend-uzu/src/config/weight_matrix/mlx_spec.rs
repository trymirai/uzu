use proc_macros::uzu_config;

use crate::config::weight_matrix::Layout;

#[uzu_config(super::WeightMatrixSpec)]
pub struct MLXSpec {
    pub bits: u32,
    pub group_size: u32,
    pub layout: Layout,
}
