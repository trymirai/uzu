use uzu_engine_macros::uzu_config;

use crate::config::{
    linear::LinearConfig,
    normalization::{NormalizationConfig, UpcastMode},
    token_mixer::convolutions::CausalConvConfig,
};

/// Scales each head vector down to `threshold` when its L2 norm is larger, after an optional learned
/// per-head per-channel scale.
#[uzu_config]
pub struct L2ClipConfig {
    pub threshold: f32,
    pub has_scale: bool,
}

/// Position-dependent scale `1 + alpha * ln(max((position + 1) / n_floor, 1))` of the queries and relative states.
#[uzu_config]
pub struct LogScalingConfig {
    pub n_floor: u32,
    pub alpha: f32,
}

/// Per-token relative states (`dim` per head, clipped) mix learned bias-versus-distance profiles into one logit
/// bias per distance; the bias is zero outside `0 <= distance < extent`.
#[uzu_config]
pub struct RelativePositionBiasConfig {
    pub dim: u32,
    pub extent: u32,
    pub clip_config: L2ClipConfig,
}

#[uzu_config(super::TokenMixerConfig)]
pub struct AttentionConfig {
    pub qkvg_projection_config: LinearConfig,
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
    pub has_qkvg_biases: bool,
    pub has_out_biases: bool,
    pub has_gate: bool,
    pub normalize_values: bool,
    pub is_kv_sharing: bool,
    /// Queries, keys and values project separate inputs. The query projection also yields the relative states,
    /// the gate and the token sinks; the key and value projections also yield their dynamic convolution taps.
    #[serde(default)]
    pub separate_inputs: bool,
    #[serde(default)]
    pub query_clip_config: Option<L2ClipConfig>,
    #[serde(default)]
    pub key_clip_config: Option<L2ClipConfig>,
    #[serde(default)]
    pub log_scaling_config: Option<LogScalingConfig>,
    #[serde(default)]
    pub relative_position_bias_config: Option<RelativePositionBiasConfig>,
    /// A sink logit per token and head, projected from the query input, with a zero value.
    #[serde(default)]
    pub has_token_sinks: bool,
    #[serde(default)]
    pub key_conv_config: Option<CausalConvConfig>,
    #[serde(default)]
    pub value_conv_config: Option<CausalConvConfig>,
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
