use proc_macros::uzu_config;

use crate::{
    config::{
        mlp::AnyMLPConfig, normalization::NormalizationConfig, per_layer_embedding::PLELayerConfig,
        rope::AnyRoPEConfig, token_mixer::AnyTokenMixerConfig,
    },
    utils::strict_serde::Unsupported,
};

#[uzu_config]
pub struct TransformerLayerConfig {
    pub pre_mixer_norm_config: Option<NormalizationConfig>,
    pub mixer_config: AnyTokenMixerConfig,
    pub post_mixer_norm_config: Option<NormalizationConfig>,
    pub pre_mlp_norm_config: NormalizationConfig,
    pub mlp_config: AnyMLPConfig,
    pub post_mlp_norm_config: Option<NormalizationConfig>,
    pub hidden_dim: Option<usize>,
    pub ple_config: Option<PLELayerConfig>,
    pub gemma4_moe_config: Option<Unsupported>,
    pub has_post_layer_scalar: bool,
    pub rope_config: Option<AnyRoPEConfig>,
    /// Newer lalamo exports declare KV sharing per layer; older bundles carry
    /// `TransformerConfig::kv_source_per_layer` instead and omit this field.
    pub kv_source_layer_index: Option<usize>,
}
