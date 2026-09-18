use uzu_engine_macros::uzu_config;

use crate::config::weight_matrix::WeightLayout;

#[uzu_config(super::WeightMatrixSpec)]
pub struct FullPrecisionSpec {
    pub layout: WeightLayout,
}
