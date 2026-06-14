use proc_macros::uzu_config;

use crate::config::{
    mlp::AnyMLPConfig, normalization::NormalizationConfig, per_layer_embedding::PLELayerConfig, rope::AnyRoPEConfig,
    token_mixer::AnyTokenMixerConfig,
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
    pub residual_moe_config: Option<AnyMLPConfig>,
    pub residual_moe_hidden_dim: Option<usize>,
    pub pre_residual_moe_norm_config: Option<NormalizationConfig>,
    pub post_dense_mlp_norm_config: Option<NormalizationConfig>,
    pub post_residual_moe_norm_config: Option<NormalizationConfig>,
    pub ple_config: Option<PLELayerConfig>,
    pub has_post_layer_scalar: bool,
    pub kv_source_layer_index: Option<usize>,
    pub rope_config: Option<AnyRoPEConfig>,
}
