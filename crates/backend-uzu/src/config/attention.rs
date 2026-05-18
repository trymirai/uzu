use proc_macros::uzu_config;

use super::{ConfigDataType, LinearConfig, NormalizationConfig, UpcastMode};

pub const VALUE_NORM_EPSILON: f32 = 1e-6;

#[uzu_config]
pub struct AttentionConfig {
    pub qkv_projection_config: LinearConfig,
    pub out_projection_config: LinearConfig,

    pub query_norm_config: Option<NormalizationConfig>,
    pub key_norm_config: Option<NormalizationConfig>,
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
    pub fn value_norm_config(&self) -> Option<NormalizationConfig> {
        if !self.normalize_values {
            return None;
        }
        Some(NormalizationConfig {
            scale_precision: self.qkv_projection_config.activation_precision(),
            accumulation_precision: ConfigDataType::Float32,
            epsilon: VALUE_NORM_EPSILON,
            scale_offset: None,
            upcast_mode: UpcastMode::FullLayer,
            subtract_mean: false,
            use_bias: false,
        })
    }
}
