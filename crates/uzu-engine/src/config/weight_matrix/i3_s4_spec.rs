use uzu_engine_macros::uzu_config;

use crate::config::weight_matrix::Layout;

/// Mirai S readout: 3-bit odd levels with a ladder scale per 64 columns, behind a 32-wide input Hadamard.
#[uzu_config(super::WeightMatrixSpec)]
pub struct I3S4Spec {
    pub layout: Layout,
}
