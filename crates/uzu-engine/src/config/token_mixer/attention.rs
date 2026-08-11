use uzu_engine_macros::uzu_config;

use crate::config::{
    linear::LinearConfig,
    normalization::{NormalizationConfig, UpcastMode},
};

#[uzu_config(super::TokenMixerConfig)]
pub struct AttentionConfig {
    pub qkv_projection_config: LinearConfig,
    pub out_projection_config: LinearConfig,

    pub query_norm_config: Option<NormalizationConfig>,
    pub key_norm_config: Option<NormalizationConfig>,

    pub num_heads: u32,
    pub num_groups: u32,
    pub head_dim: u32,
    pub is_causal: bool,
    pub scale: Option<f32>,
    pub sliding_window_size: Option<u32>,
    pub logit_soft_cap: Option<f32>,
    pub has_sinks: bool,
    pub has_qkv_biases: bool,
    pub has_out_biases: bool,
    pub has_gate: bool,
    pub normalize_values: bool,
    pub is_kv_sharing: bool,
}

impl AttentionConfig {
    pub fn value_norm_config(&self) -> Option<NormalizationConfig> {
        self.normalize_values.then_some(NormalizationConfig {
            epsilon: 1e-6,
            scale_offset: None,
            upcast_mode: UpcastMode::FullLayer,
            subtract_mean: false,
            has_scale: false,
            has_biases: false,
        })
    }
}
