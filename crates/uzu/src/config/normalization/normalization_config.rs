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
    #[serde(default)]
    pub subtract_mean: bool,
}

#[cfg(test)]
#[path = "../../../tests_unit/config/normalization/normalization_config_test.rs"]
mod tests;
