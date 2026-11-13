use uzu_engine_macros::uzu_config;

use super::Layout;
use crate::backends::common::gpu_types::trellis::TrellisCodebook;

#[uzu_config(super::WeightMatrixSpec)]
pub struct QtipGaussianSpec {
    pub vector_width: u32,
    pub transition_bits: u32,
    pub restart_columns: u32,
    pub codebook: TrellisCodebook,
    pub layout: Layout,
    pub tape: monostate::MustBe!(2u64),
}
