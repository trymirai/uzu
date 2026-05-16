use proc_macros::uzu_config;

use super::UpcastMode;
use crate::ConfigDataType;

#[uzu_config]
pub struct NormalizationConfig {
    pub scale_precision: ConfigDataType,
    pub accumulation_precision: ConfigDataType,
    pub epsilon: f32,
    pub scale_offset: Option<f32>,
    pub upcast_mode: UpcastMode,
    pub subtract_mean: bool,
    pub use_bias: bool,
}
