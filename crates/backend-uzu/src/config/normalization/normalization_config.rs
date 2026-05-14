use serde::{Deserialize, Serialize};

use super::UpcastMode;
use crate::ConfigDataType;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct NormalizationConfig {
    pub scale_precision: ConfigDataType,
    pub accumulation_precision: ConfigDataType,
    pub epsilon: f32,
    pub scale_offset: Option<f32>,
    pub upcast_mode: UpcastMode,
    pub subtract_mean: bool,
    pub use_bias: bool,
}
