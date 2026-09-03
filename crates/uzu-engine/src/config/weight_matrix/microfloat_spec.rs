use uzu_engine_macros::uzu_config;

use crate::{backends::common::microfloat::MicrofloatFormat, config::weight_matrix::Layout};

#[uzu_config(super::WeightMatrixSpec)]
pub struct MicrofloatSpec {
    pub bits: u32,
    pub group_size: usize,
    pub scale_mode: MicrofloatFormat,
    pub layout: Layout,
}
