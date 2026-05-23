use proc_macros::uzu_config;

use super::UpcastMode;
use crate::DataType;

#[uzu_config]
pub struct NormalizationConfig {
    pub scale_precision: DataType,
    pub accumulation_precision: DataType,
    pub epsilon: f32,
    pub scale_offset: Option<f32>,
    pub upcast_mode: UpcastMode,
    pub subtract_mean: bool,
    pub use_bias: bool,
}
