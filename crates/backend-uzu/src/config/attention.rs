use serde::{Deserialize, Serialize};

use super::{ConfigDataType, LinearConfig, NormalizationConfig, UpcastMode};

pub const VALUE_NORM_EPSILON: f32 = 1e-6;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct AttentionConfig {
    pub qkv_projection_config: LinearConfig,
    pub out_projection_config: LinearConfig,

    pub query_norm_config: Option<NormalizationConfig>,
    pub key_norm_config: Option<NormalizationConfig>,
    #[serde(default)]
    pub normalize_values: bool,

    pub num_heads: usize,
    pub num_groups: usize,
    pub head_dim: usize,
    pub is_causal: bool,
    pub scale: Option<f32>,
    pub sliding_window_size: Option<usize>,

    pub logit_soft_cap: Option<f32>,
    pub has_sinks: bool,
    pub has_qkv_biases: bool,
    pub has_out_biases: bool,
    pub gate_projection_config: Option<LinearConfig>,
}

impl AttentionConfig {
    /// Scale-free RMSNorm on values; precision mirrors query/key norm.
    pub fn value_norm_config(&self) -> Option<NormalizationConfig> {
        if !self.normalize_values {
            return None;
        }
        let base = self.query_norm_config.as_ref().or(self.key_norm_config.as_ref());
        Some(NormalizationConfig {
            scale_precision: base.map_or(ConfigDataType::Float32, |norm| norm.scale_precision),
            accumulation_precision: base.map_or(ConfigDataType::Float32, |norm| norm.accumulation_precision),
            epsilon: VALUE_NORM_EPSILON,
            scale_offset: None,
            upcast_mode: base.map_or(UpcastMode::FullLayer, |norm| norm.upcast_mode.clone()),
            subtract_mean: false,
            use_bias: false,
        })
    }
}
