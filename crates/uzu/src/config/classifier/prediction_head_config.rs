use serde::{Deserialize, Serialize};

use crate::{LinearConfig, NormalizationConfig, backends::common::ActivationConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct PredictionHeadConfig {
    pub dense_config: LinearConfig,
    pub activation: ActivationConfig,
    pub normalization_config: NormalizationConfig,
    pub readout_config: LinearConfig,
    pub use_dense_bias: bool,
}
