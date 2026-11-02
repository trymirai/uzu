use uzu_engine_macros::uzu_config;

use crate::config::weight_matrix::Layout;

/// Lookup-only D4S4 table: one D4 lattice code per 4 columns and a ladder scale per 64 columns.
#[uzu_config(super::WeightMatrixSpec)]
pub struct D4S4Spec {
    pub layout: Layout,
}

/// Lattice weights exported by Lalamo.
#[uzu_config(super::WeightMatrixSpec)]
pub struct LatticeSpec {
    pub kind: monostate::MustBe!("d4"),
    pub layout: Layout,
}
