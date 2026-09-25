use uzu_engine_macros::uzu_config;

use crate::config::weight_matrix::Layout;

/// Lookup-only Mirai S embedding table: one D4 lattice code per 4 columns and a ladder scale per 64 columns.
#[uzu_config(super::WeightMatrixSpec)]
pub struct D4S4Spec {
    pub layout: Layout,
}
