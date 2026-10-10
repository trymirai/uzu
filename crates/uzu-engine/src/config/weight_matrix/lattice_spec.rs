use uzu_engine_macros::uzu_config;

use crate::config::weight_matrix::Layout;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum LatticeKind {
    D4,
}

#[uzu_config(super::WeightMatrixSpec)]
pub struct LatticeSpec {
    pub kind: LatticeKind,
    pub layout: Layout,
}
