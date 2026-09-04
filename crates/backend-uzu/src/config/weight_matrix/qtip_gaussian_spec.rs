use proc_macros::uzu_config;

use crate::config::weight_matrix::Layout;

#[uzu_config(super::WeightMatrixSpec)]
pub struct QtipGaussianSpec {
    pub layout: Layout,
    pub vector_width: u32,
    pub transition_bits: u32,
    pub restart_columns: u32,
}
