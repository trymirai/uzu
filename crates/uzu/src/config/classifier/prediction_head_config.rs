use serde::{Deserialize, Serialize};

use crate::{Activation, LinearConfig, NormalizationConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct PredictionHeadConfig {
    pub dense_config: LinearConfig,
    pub activation: Activation,
    pub normalization_config: NormalizationConfig,
    pub readout_config: LinearConfig,
    pub use_dense_bias: bool,
}
