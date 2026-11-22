use uzu_engine_macros::uzu_config;

use crate::config::{
    linear::LinearConfig,
    mlp::AnyMLPConfig,
    normalization::NormalizationConfig,
    rope::AnyRoPEConfig,
    stream_op::AnyStreamOpConfig,
    token_mixer::{
        AnyTokenMixerConfig,
        convolutions::{CausalConvConfig, SeparableCausalConvConfig},
    },
};

#[uzu_config]
pub struct TransformerLayerConvConfig {
    pub conv_config: SeparableCausalConvConfig,
    pub kernel_projection_config: LinearConfig,
    pub conv_kernel_size: u32,
    pub conv_group_size: u32,
}

fn single_input() -> u32 {
    1
}

#[uzu_config]
pub struct SublayerConfigCommon {
    pub pre_norm_config: Option<NormalizationConfig>,
    pub conv_config: Option<TransformerLayerConvConfig>,
    pub post_norm_config: Option<NormalizationConfig>,
    /// The body takes this many inputs, e.g. separate query, key and value inputs. They share one read of the
    /// residual stream and one pre-norm unless `separate_input_norms` is set.
    #[serde(default = "single_input")]
    pub num_inputs: u32,
    #[serde(default)]
    pub separate_input_norms: bool,
    /// A stateful convolution on the body outputs, before the post-norm.
    #[serde(default)]
    pub output_conv_config: Option<CausalConvConfig>,
}

impl SublayerConfigCommon {
    /// How many reads of the residual stream the inputs take: one shared read unless the inputs are normalized
    /// separately.
    pub fn num_reads(&self) -> u32 {
        if self.separate_input_norms { self.num_inputs } else { 1 }
    }
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
