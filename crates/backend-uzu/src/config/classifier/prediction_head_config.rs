use proc_macros::uzu_config;

use crate::{
    backends::common::ActivationConfig,
    config::{LinearConfig, NormalizationConfig},
};

#[uzu_config]
pub struct PredictionHeadConfig {
    pub dense_config: LinearConfig,
    pub activation: ActivationConfig,
    pub normalization_config: NormalizationConfig,
    pub readout_config: LinearConfig,
    pub use_dense_bias: bool,
}
