use uzu_engine_macros::uzu_config;

use crate::config::weight_matrix::{QuantParamsLayout, WeightLayout};

#[uzu_config(super::WeightMatrixSpec)]
pub struct IntSpec {
    pub bits: u32,
    pub group_size: u32,
    pub is_symmetric: bool,
    pub weight_layout: WeightLayout,
    pub params_layout: QuantParamsLayout,
}
