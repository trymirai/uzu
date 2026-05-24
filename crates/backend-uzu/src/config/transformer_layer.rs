use proc_macros::uzu_config;

use super::{MLPConfig, MixerConfig, NormalizationConfig, PLELayerConfig, RoPEConfig};

#[uzu_config]
pub struct TransformerLayerConfig {
    pub pre_mixer_norm_config: Option<NormalizationConfig>,
    pub mixer_config: MixerConfig,
    pub post_mixer_norm_config: Option<NormalizationConfig>,
    pub pre_mlp_norm_config: NormalizationConfig,
    pub mlp_config: MLPConfig,
    pub post_mlp_norm_config: Option<NormalizationConfig>,
    pub rope_config: Option<RoPEConfig>,
    pub hidden_dim: Option<usize>,
    pub ple_config: Option<PLELayerConfig>,
    pub has_post_layer_scalar: bool,
    pub kv_source_layer: Option<usize>,
}
