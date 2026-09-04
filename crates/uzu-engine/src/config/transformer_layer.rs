use uzu_engine_macros::uzu_config;

use crate::config::{
    linear::LinearConfig, mlp::AnyMLPConfig, normalization::NormalizationConfig, per_layer_embedding::PLELayerConfig,
    rope::AnyRoPEConfig, token_mixer::AnyTokenMixerConfig,
};

#[uzu_config]
pub struct GroupedConvolutionConfig {
    pub kernel_size: u32,
    pub group_size: u32,
    pub kernel_projection_config: LinearConfig,
}

#[uzu_config]
pub struct TransformerLayerConfig {
    pub pre_mixer_norm_config: Option<NormalizationConfig>,
    pub mixer_config: AnyTokenMixerConfig,
    pub post_mixer_norm_config: Option<NormalizationConfig>,
    pub pre_mlp_norm_config: NormalizationConfig,
    pub mlp_config: AnyMLPConfig,
    pub post_mlp_norm_config: Option<NormalizationConfig>,
    pub hidden_dim: Option<u32>,
    pub ple_config: Option<PLELayerConfig>,
    pub has_post_layer_scalar: bool,
    pub kv_source_layer_index: Option<u32>,
    pub rope_config: Option<AnyRoPEConfig>,
    #[serde(default)]
    pub grouped_convolution_config: Option<GroupedConvolutionConfig>,
}
