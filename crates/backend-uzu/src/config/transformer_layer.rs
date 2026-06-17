use proc_macros::uzu_config;

use crate::config::{
    mlp::{AnyMLPConfig, mixture_of_experts::MixtureOfExpertsConfig},
    normalization::NormalizationConfig,
    per_layer_embedding::PLELayerConfig,
    rope::AnyRoPEConfig,
    token_mixer::AnyTokenMixerConfig,
};

#[uzu_config]
pub struct Gemma4MoEBlockConfig {
    pub moe_config: MixtureOfExpertsConfig,
    pub norm_config: NormalizationConfig,
    pub router_norm_epsilon: f32,
}

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
    pub gemma4_moe_config: Option<Gemma4MoEBlockConfig>,
    pub has_post_layer_scalar: bool,
    pub rope_config: Option<AnyRoPEConfig>,
    pub kv_source_layer_index: Option<usize>,
}
