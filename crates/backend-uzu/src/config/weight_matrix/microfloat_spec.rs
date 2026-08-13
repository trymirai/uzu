use proc_macros::uzu_config;

use crate::config::weight_matrix::Layout;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum MicrofloatScaleMode {
    Mxfp4,
    Nvfp4,
}

#[uzu_config(super::WeightMatrixSpec)]
pub struct MicrofloatSpec {
    pub bits: u32,
    pub group_size: usize,
    pub scale_mode: MicrofloatScaleMode,
    pub layout: Layout,
}
