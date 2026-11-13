use uzu_engine_macros::uzu_config;

use crate::config::{
    linear::LinearConfig,
    mlp::AnyMLPConfig,
    normalization::NormalizationConfig,
    rope::AnyRoPEConfig,
    stream_op::AnyStreamOpConfig,
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
pub struct SublayerConfigCommon {
    pub pre_norm_config: Option<NormalizationConfig>,
    pub conv_config: Option<TransformerLayerConvConfig>,
    pub post_norm_config: Option<NormalizationConfig>,
}

#[uzu_config]
pub struct MixerSublayerConfig {
    #[serde(flatten)]
    pub common: SublayerConfigCommon,
    pub body_config: AnyTokenMixerConfig,
}

#[uzu_config]
pub struct MlpSublayerConfig {
    #[serde(flatten)]
    pub common: SublayerConfigCommon,
    pub body_config: AnyMLPConfig,
}

#[uzu_config]
pub struct TransformerLayerConfig {
    pub prologue_configs: Box<[AnyStreamOpConfig]>,
    pub mixer: MixerSublayerConfig,
    pub mlp: MlpSublayerConfig,
    pub epilogue_configs: Box<[AnyStreamOpConfig]>,
    pub hidden_dim: Option<u32>,
    pub kv_source_layer_index: Option<u32>,
    pub rope_config: Option<AnyRoPEConfig>,
}
