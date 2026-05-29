use monostate::MustBe;
use proc_macros::uzu_config;

use crate::config::{
    activation::AnyActivation,
    embedding::tied_embedding::TiedEmbeddingConfig,
    linear::LinearConfig,
    normalization::NormalizationConfig,
    transformer::TransformerConfig,
    tts::audio_decoder::common::{CausalConv1dConfig, CausalTransposeConv1dConfig, Snake1dConfig},
};

#[uzu_config]
pub struct ConvNeXtBlockConfig {
    pub activation: AnyActivation,
    pub dwconv_config: CausalConv1dConfig,
    pub norm_config: NormalizationConfig,
    pub pwconv_config: LinearConfig,
}

#[uzu_config]
pub struct UpsamplingBlockConfig {
    pub trans_conv_config: CausalTransposeConv1dConfig,
    pub convnext_config: ConvNeXtBlockConfig,
}

#[uzu_config]
pub struct UpsamplerConfig {
    pub block_configs: Box<[UpsamplingBlockConfig]>,
}

#[uzu_config]
pub struct VectorQuantizeConfig {
    pub codebook_config: TiedEmbeddingConfig,
    pub out_proj_config: LinearConfig,
}

#[uzu_config]
pub struct ResidualVectorQuantizeConfig {
    pub vq_config: VectorQuantizeConfig,
}

#[uzu_config]
pub struct DownsampleResidualVectorQuantizeConfig {
    pub semantic_quantizer_config: ResidualVectorQuantizeConfig,
    pub quantizer_config: ResidualVectorQuantizeConfig,
    pub post_module_config: TransformerConfig,
    pub upsampler_config: UpsamplerConfig,
}

#[uzu_config]
pub struct ResidualUnitConfig {
    pub snake_config: Snake1dConfig,
    pub conv_config: CausalConv1dConfig,
    pub causal: MustBe!(true),
}

#[uzu_config]
pub struct DACDecoderBlockConfig {
    pub snake_config: Snake1dConfig,
    pub trans_conv_config: CausalTransposeConv1dConfig,
    pub res_unit_config: ResidualUnitConfig,
    pub causal: MustBe!(true),
}

#[uzu_config]
pub struct DACDecoderConfig {
    pub conv_config: CausalConv1dConfig,
    pub snake_config: Snake1dConfig,
    pub decoder_block_config: DACDecoderBlockConfig,
    pub causal: MustBe!(true),
}
