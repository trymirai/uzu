use backend_uzu_macros::uzu_config;

use crate::config::weight_matrix::Layout;

#[uzu_config(super::WeightMatrixSpec)]
pub struct IntSpec {
    pub bits: u32,
    pub group_size: u32,
    pub is_symmetric: bool,
    pub layout: Layout,
}
