use uzu_engine_macros::uzu_config;

use crate::config::weight_matrix::Layout;

/// 2-bit symmetric per-row weights, BF16 row scales.
#[uzu_config(super::WeightMatrixSpec)]
pub struct SymRow2Spec {
    pub layout: Layout,
}
