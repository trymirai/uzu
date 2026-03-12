use std::{collections::BTreeMap, path::Path};

use serde::{Deserialize, Serialize};

use crate::{ConfigDataType, EmbeddingConfig, EmbeddingConfigCommon, LinearConfig, TransformerConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct TtsMessageProcessorConfig {
    pub prompt_template: String,
    #[serde(default = "default_drop_initial_newline")]
    pub drop_initial_newline: bool,
}

fn default_drop_initial_newline() -> bool {
    true
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone, Default)]
pub struct TtsVocoderConfig {
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct TtsConfig {
    pub text_decoder_config: TtsTextDecoderConfig,
    pub audio_decoder_config: TtsAudioDecoderConfig,
    pub vocoder_config: TtsVocoderConfig,
    #[serde(default)]
    pub activation_precision: Option<ConfigDataType>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum TtsTextDecoderConfig {
    StubTextDecoderConfig {
        num_codebooks: usize,
        codebook_size: usize,
    },
    FishAudioTextDecoderConfig {
        #[serde(flatten)]
        config: FishAudioTextDecoderConfig,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct FishAudioTextDecoderConfig {
    pub slow_embeddings_config: FishAudioEmbeddingConfig,
    pub slow_model_config: TransformerConfig,
    pub slow_readout_config: FishAudioLinearConfig,
    pub fast_embeddings_config: FishAudioEmbeddingConfig,
    pub fast_model_config: TransformerConfig,
    pub fast_readout_config: FishAudioLinearConfig,
    pub codebook_embeddings_config: FishAudioEmbeddingConfig,
    pub fast_model_projection_config: Option<FishAudioLinearConfig>,
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
    #[serde(default)]
    pub precision: Option<ConfigDataType>,
    pub short_logits_size: usize,
    pub repeat_window_size: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(untagged)]
pub enum FishAudioLinearConfig {
    Tagged(LinearConfig),
    UntaggedFullPrecision {
        #[serde(rename = "precision")]
        precision: ConfigDataType,
    },
}

impl FishAudioLinearConfig {
    pub fn is_full_precision(&self) -> bool {
        matches!(
            self,
            FishAudioLinearConfig::Tagged(LinearConfig::FullPrecision { .. })
                | FishAudioLinearConfig::UntaggedFullPrecision { .. }
        )
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(untagged)]
pub enum FishAudioEmbeddingConfig {
    Tagged(EmbeddingConfig),
    UntaggedFullPrecision {
        input_scale: Option<f32>,
        logit_soft_cap: Option<f32>,
        precision: ConfigDataType,
    },
}

impl FishAudioEmbeddingConfig {
    pub fn to_embedding_config(&self) -> EmbeddingConfig {
        match self {
            FishAudioEmbeddingConfig::Tagged(config) => config.clone(),
            FishAudioEmbeddingConfig::UntaggedFullPrecision {
                input_scale,
                logit_soft_cap,
                precision,
            } => EmbeddingConfig::Untied {
                common: EmbeddingConfigCommon {
                    input_scale: *input_scale,
                    logit_soft_cap: *logit_soft_cap,
                },
                precision: *precision,
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum TtsAudioDecoderConfig {
    NanoCodecConfig {
        #[serde(flatten)]
        config: NanoCodecAudioDecoderConfig,
    },
    DescriptAudioCodecConfig {
        #[serde(flatten)]
        config: DescriptAudioCodecConfig,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct NanoCodecAudioDecoderConfig {
    pub samplerate: u32,
    pub quantizer_config: NanoCodecGroupedQuantizerConfig,
    pub decoder_config: NanoCodecDecoderConfig,
    pub base_channels: usize,
    pub up_sample_rates: Vec<usize>,
    #[serde(default)]
    pub in_kernel_size: Option<usize>,
    #[serde(default)]
    pub out_kernel_size: Option<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilations: Vec<usize>,
    #[serde(default)]
    pub precision: Option<ConfigDataType>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct NanoCodecGroupedQuantizerConfig {
    pub num_groups: usize,
    pub quantizer_config: NanoCodecQuantizerConfig,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct NanoCodecQuantizerConfig {
    pub num_levels: Vec<i32>,
    #[serde(default)]
    pub eps: Option<f32>,
    #[serde(default)]
    pub precision: Option<ConfigDataType>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct NanoCodecDecoderConfig {
    pub activation_config: NanoCodecActivationConfig,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct NanoCodecActivationConfig {
    #[serde(default)]
    pub leaky_relu_negative_slope: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DescriptAudioCodecConfig {
    pub precision: ConfigDataType,
    pub quantizer_config: DescriptAudioQuantizerConfig,
    pub decoder_config: serde_json::Value,
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

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DescriptAudioQuantizerConfig {
    #[serde(default)]
    pub precision: Option<ConfigDataType>,
    pub post_module_config: TransformerConfig,
    pub upsampler_config: DescriptAudioUpsamplerConfig,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DescriptAudioUpsamplerConfig {
    pub block_configs: Vec<DescriptAudioUpsamplingBlockConfig>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DescriptAudioUpsamplingBlockConfig {
    pub convnext_config: DescriptAudioConvNeXtConfig,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DescriptAudioConvNeXtConfig {
    pub norm_config: DescriptAudioConvNeXtNormConfig,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DescriptAudioConvNeXtNormConfig {
    pub epsilon: f32,
    #[serde(default)]
    pub subtract_mean: bool,
    #[serde(default)]
    pub use_bias: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct TtsModelConfig {
    pub tts_config: TtsConfig,
    pub message_processor_config: TtsMessageProcessorConfig,
}

#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
impl TtsModelConfig {
    pub fn create_audio_generation_context(&self) -> crate::audio::AudioResult<crate::audio::AudioGenerationContext> {
        crate::audio::AudioGenerationContext::from_tts_config(&self.tts_config)
    }

    pub fn create_audio_generation_context_with_model_path(
        &self,
        model_path: &Path,
    ) -> crate::audio::AudioResult<crate::audio::AudioGenerationContext> {
        crate::audio::AudioGenerationContext::from_tts_config_and_model_path(&self.tts_config, model_path)
    }

    pub fn create_audio_generation_context_with_model_path_and_options(
        &self,
        model_path: &Path,
        options: crate::audio::NanoCodecFsqRuntimeOptions,
    ) -> crate::audio::AudioResult<crate::audio::AudioGenerationContext> {
        crate::audio::AudioGenerationContext::from_tts_config_and_model_path_with_options(
            &self.tts_config,
            model_path,
            options,
        )
    }

    pub fn create_audio_codec_runtime(&self) -> crate::audio::AudioResult<crate::audio::NanoCodecFsqRuntime> {
        crate::audio::NanoCodecFsqRuntime::from_tts_config(&self.tts_config)
    }

    pub fn create_audio_codec_runtime_with_model_path(
        &self,
        model_path: &Path,
    ) -> crate::audio::AudioResult<crate::audio::NanoCodecFsqRuntime> {
        crate::audio::NanoCodecFsqRuntime::from_tts_config_and_model_path(&self.tts_config, model_path)
    }

    pub fn create_audio_codec_runtime_with_model_path_and_options(
        &self,
        model_path: &Path,
        options: crate::audio::NanoCodecFsqRuntimeOptions,
    ) -> crate::audio::AudioResult<crate::audio::NanoCodecFsqRuntime> {
        crate::audio::NanoCodecFsqRuntime::from_tts_config_and_model_path_with_options(
            &self.tts_config,
            model_path,
            options,
        )
    }
}
