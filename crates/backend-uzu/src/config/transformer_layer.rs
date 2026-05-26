use proc_macros::uzu_config;

use crate::{
    config::{
        mlp::AnyMLPConfig, normalization::NormalizationConfig, rope::AnyRoPEConfig, token_mixer::AnyTokenMixerConfig,
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
    pub ple_config: Option<Unsupported>,
    pub has_post_layer_scalar: bool,
    pub kv_source_layer_index: Option<usize>,
    pub rope_config: Option<AnyRoPEConfig>,
}
