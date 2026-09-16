use uzu_engine_macros::uzu_config;

use crate::config::{
    linear::LinearConfig,
    mlp::AnyMLPConfig,
    normalization::NormalizationConfig,
    per_layer_embedding::PLELayerConfig,
    rope::AnyRoPEConfig,
    token_mixer::{AnyTokenMixerConfig, convolutions::SeparableCausalConvConfig},
};

#[uzu_config]
pub struct TransformerLayerConvConfig {
    pub conv_config: SeparableCausalConvConfig,
    pub kernel_projection_config: LinearConfig,
    pub conv_kernel_size: u32,
    pub conv_group_size: u32,
}

#[uzu_config]
pub struct TransformerLayerConfig {
    pub pre_mixer_norm_config: Option<NormalizationConfig>,
    pub mixer_config: AnyTokenMixerConfig,
    pub mixer_conv_config: Option<TransformerLayerConvConfig>,
    pub post_mixer_norm_config: Option<NormalizationConfig>,
    pub pre_mlp_norm_config: NormalizationConfig,
    pub mlp_config: AnyMLPConfig,
    pub mlp_conv_config: Option<TransformerLayerConvConfig>,
    pub post_mlp_norm_config: Option<NormalizationConfig>,
    pub hidden_dim: Option<u32>,
    pub ple_config: Option<PLELayerConfig>,
    pub has_post_layer_scalar: bool,
    pub kv_source_layer_index: Option<u32>,
    pub rope_config: Option<AnyRoPEConfig>,
}
