use uzu_engine_macros::uzu_config;

use crate::config::{
    linear::LinearConfig,
    normalization::{NormalizationConfig, UpcastMode},
    token_mixer::convolutions::SeparableCausalConvConfig,
};

#[uzu_config]
pub struct L2ClipConfig {
    pub threshold: f32,
    pub epsilon: f32,
}

// Not explicitly named in lalamo
#[uzu_config]
#[serde(untagged)]
pub enum HeadNormConfig {
    Normalization(NormalizationConfig),
    L2Clip(L2ClipConfig),
}

#[uzu_config]
pub struct RelativeBiasConfig {
    pub num_channels: u32,
    pub extent: u32,
    pub clip_config: L2ClipConfig,
}

#[uzu_config]
pub struct LogScalingConfig {
    pub alpha: f32,
    pub position_floor: u32,
}

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum ProjectionLayout {
    StackedQueryFirst,
    GroupedWithConvTaps,
}

#[uzu_config]
pub struct KeyValueConvConfig {
    pub conv_config: SeparableCausalConvConfig,
    pub kernel_size: u32,
}

#[uzu_config]
pub struct StackedProjectionConfig {
    pub layout: ProjectionLayout,
    pub relative_bias_config: RelativeBiasConfig,
    pub key_value_conv_config: KeyValueConvConfig,
}

#[uzu_config(super::TokenMixerConfig)]
pub struct AttentionConfig {
    pub qkvg_projection_config: LinearConfig,
    pub out_projection_config: LinearConfig,

    pub query_norm_config: Option<HeadNormConfig>,
    pub key_norm_config: Option<HeadNormConfig>,

    pub num_heads: u32,
    pub num_groups: u32,
    pub head_dim: u32,
    pub is_causal: bool,
    pub scale: Option<f32>,
    pub sliding_window_size: Option<u32>,
    pub logit_soft_cap: Option<f32>,
    pub has_sinks: bool,
    pub has_qkvg_biases: bool,
    pub has_out_biases: bool,
    pub has_gate: bool,
    pub normalize_values: bool,
    pub is_kv_sharing: bool,
    pub log_scaling_config: Option<LogScalingConfig>,
    pub stacked_projection_config: Option<StackedProjectionConfig>,
}

impl AttentionConfig {
    pub fn qkvg_head_count(&self) -> u32 {
        let mut heads = self.num_heads; // query
        if !self.is_kv_sharing {
            heads += 2 * self.num_groups; // key + value
        }
        if self.has_gate {
            heads += self.num_heads; // gate
        }
        heads
    }

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
