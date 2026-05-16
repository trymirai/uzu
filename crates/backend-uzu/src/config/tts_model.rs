#[cfg(metal_backend)]
use std::path::Path;

use proc_macros::uzu_config;

use super::{EmbeddingConfig, EmbeddingConfigCommon, NormalizationConfig, TransformerConfig};
use crate::{ConfigDataType, backends::common::ActivationConfig};

#[uzu_config]
pub struct TtsMessageProcessorConfig {
    pub prompt_template: String,
    pub drop_initial_newline: bool,
}

#[uzu_config]
pub struct TtsConfig {
    pub text_decoder_config: TtsTextDecoderConfig,
    pub audio_decoder_config: TtsAudioDecoderConfig,
    pub vocoder_config: NoopVocoderConfig,
    pub activation_precision: ConfigDataType,
}

#[uzu_config]
pub struct NoopVocoderConfig {}

#[uzu_config]
#[serde(tag = "type")]
pub enum TtsTextDecoderConfig {
    #[serde(rename = "FishAudioTextDecoderConfig")]
    FishAudio(FishAudioTextDecoderConfig),
}

#[uzu_config]
pub struct FishAudioTextDecoderConfig {
    pub slow_embeddings_config: TiedEmbeddingConfig,
    pub slow_model_config: TransformerConfig,
    pub slow_readout_config: FullPrecisionLinearConfig,
    pub fast_embeddings_config: TiedEmbeddingConfig,
    pub fast_model_config: TransformerConfig,
    pub fast_readout_config: FullPrecisionLinearConfig,
    pub codebook_embeddings_config: TiedEmbeddingConfig,
    pub fast_model_projection_config: Option<FullPrecisionLinearConfig>,
    pub semantic_token_begin_id: i64,
    pub semantic_token_end_id: i64,
    pub im_end_token_id: i64,
    pub codebook_size: usize,
    pub vocab_size: usize,
    pub slow_model_dim: usize,
    pub fast_model_dim: usize,
    pub num_codebooks: usize,
    pub max_seq_len: usize,
    pub scale_codebook_embeddings: bool,
    pub precision: ConfigDataType,
    pub short_logits_size: usize,
    pub repeat_window_size: usize,
}

#[uzu_config]
pub struct FullPrecisionLinearConfig {
    pub precision: ConfigDataType,
}

#[uzu_config]
pub struct TiedEmbeddingConfig {
    #[serde(flatten)]
    pub common: EmbeddingConfigCommon,
    pub precision: ConfigDataType,
}

impl TiedEmbeddingConfig {
    #[cfg(metal_backend)]
    pub(crate) fn to_text_decoder_embedding_config(&self) -> EmbeddingConfig {
        EmbeddingConfig::Untied {
            common: self.common.clone(),
            precision: self.precision,
        }
    }
}

#[uzu_config]
#[serde(tag = "type")]
pub enum TtsAudioDecoderConfig {
    #[serde(rename = "DescriptAudioCodecConfig")]
    DescriptAudioCodec(DescriptAudioCodecConfig),
}

#[uzu_config]
pub struct DescriptAudioCodecConfig {
    pub precision: ConfigDataType,
    pub quantizer_config: DescriptAudioQuantizerConfig,
    pub decoder_config: DescriptAudioDacDecoderConfig,
    pub samplerate: u32,
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub input_dim: usize,
    pub n_codebooks: usize,
    pub codebook_dim: usize,
    pub downsample_factor: Vec<usize>,
    pub codebook_size: usize,
    pub semantic_codebook_size: usize,
}

#[uzu_config]
pub struct DescriptAudioDacDecoderConfig {
    pub precision: ConfigDataType,
    pub conv_config: DescriptAudioCausalConv1dConfig,
    pub snake_config: DescriptAudioSnake1dConfig,
    pub decoder_block_config: DescriptAudioDacDecoderBlockConfig,
    pub causal: bool,
}

#[uzu_config]
pub struct DescriptAudioDacDecoderBlockConfig {
    pub precision: ConfigDataType,
    pub snake_config: DescriptAudioSnake1dConfig,
    pub trans_conv_config: DescriptAudioCausalTransposeConv1dConfig,
    pub res_unit_config: DescriptAudioResidualUnitConfig,
    pub causal: bool,
}

#[uzu_config]
pub struct DescriptAudioResidualUnitConfig {
    pub precision: ConfigDataType,
    pub snake_config: DescriptAudioSnake1dConfig,
    pub conv_config: DescriptAudioCausalConv1dConfig,
    pub causal: bool,
}

#[uzu_config]
pub struct DescriptAudioCausalConv1dConfig {
    pub precision: ConfigDataType,
    pub has_biases: bool,
}

#[uzu_config]
pub struct DescriptAudioCausalTransposeConv1dConfig {
    pub precision: ConfigDataType,
    pub has_biases: bool,
}

#[uzu_config]
pub struct DescriptAudioSnake1dConfig {
    pub precision: ConfigDataType,
}

#[uzu_config]
pub struct DescriptAudioQuantizerConfig {
    pub precision: ConfigDataType,
    pub semantic_quantizer_config: DescriptAudioResidualVectorQuantizeConfig,
    pub quantizer_config: DescriptAudioResidualVectorQuantizeConfig,
    pub post_module_config: TransformerConfig,
    pub upsampler_config: DescriptAudioUpsamplerConfig,
}

#[uzu_config]
pub struct DescriptAudioResidualVectorQuantizeConfig {
    pub precision: ConfigDataType,
    pub vq_config: DescriptAudioVectorQuantizeConfig,
}

#[uzu_config]
pub struct DescriptAudioVectorQuantizeConfig {
    pub precision: ConfigDataType,
    pub codebook_config: TiedEmbeddingConfig,
    pub out_proj_config: FullPrecisionLinearConfig,
}

#[uzu_config]
pub struct DescriptAudioUpsamplerConfig {
    pub block_configs: Vec<DescriptAudioUpsamplingBlockConfig>,
}

#[uzu_config]
pub struct DescriptAudioUpsamplingBlockConfig {
    pub precision: ConfigDataType,
    pub trans_conv_config: DescriptAudioCausalTransposeConv1dConfig,
    pub convnext_config: DescriptAudioConvNeXtConfig,
}

#[uzu_config]
pub struct DescriptAudioConvNeXtConfig {
    pub precision: ConfigDataType,
    pub activation: ActivationConfig,
    pub dwconv_config: DescriptAudioCausalConv1dConfig,
    pub norm_config: NormalizationConfig,
    pub pwconv_config: FullPrecisionLinearConfig,
}

#[uzu_config]
pub struct TtsModelConfig {
    pub tts_config: TtsConfig,
    pub message_processor_config: TtsMessageProcessorConfig,
}

#[cfg(metal_backend)]
impl TtsModelConfig {
    pub fn create_audio_generation_context_with_model_path<B: crate::backends::common::Backend>(
        &self,
        model_path: &Path,
    ) -> crate::audio::AudioResult<crate::audio::AudioGenerationContext<B>> {
        crate::audio::AudioGenerationContext::from_tts_config_and_model_path(&self.tts_config, model_path)
    }
}
