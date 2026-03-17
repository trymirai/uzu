use serde::{Deserialize, Serialize};

use crate::{LinearConfig, NormalizationConfig, backends::common::gpu_types::ActivationType};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct PredictionHeadConfig {
    pub dense_config: LinearConfig,
    pub activation: ActivationType,
    pub normalization_config: NormalizationConfig,
    pub readout_config: LinearConfig,
    pub use_dense_bias: bool,
}
