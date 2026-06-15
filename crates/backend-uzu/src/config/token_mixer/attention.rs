use proc_macros::uzu_config;

use crate::config::{
    linear::LinearConfig,
    normalization::{NormalizationConfig, UpcastMode},
};

#[uzu_config]
#[derive(Default)]
#[serde(rename_all = "snake_case")]
pub enum AttentionProjectionMode {
    #[default]
    Qkv,
    QkSharedValue,
    BorrowedQ,
}

#[uzu_config(super::TokenMixerConfig)]
pub struct AttentionConfig {
    pub qkv_projection_config: LinearConfig,
    pub out_projection_config: LinearConfig,

    pub query_norm_config: Option<NormalizationConfig>,
    pub key_norm_config: Option<NormalizationConfig>,

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
    pub normalize_values: bool,
    /// Newer lalamo exports omit this; fused QKV is the standard layout.
    pub projection_mode: Option<AttentionProjectionMode>,
    /// Newer lalamo exports mark KV-sharing producer layers; consumers carry
    /// `kv_source_layer_index`, which is what the runtime keys off.
    pub is_kv_sharing: Option<bool>,
}

impl AttentionConfig {
    pub fn projection_mode(&self) -> AttentionProjectionMode {
        self.projection_mode.clone().unwrap_or_default()
    }

    pub fn value_norm_config(&self) -> Option<NormalizationConfig> {
        self.normalize_values.then_some(NormalizationConfig {
            epsilon: 1e-6,
            scale_offset: None,
            upcast_mode: UpcastMode::FullLayer,
            subtract_mean: false,
            has_biases: false,
        })
    }
}
