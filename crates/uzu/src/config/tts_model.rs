use std::{collections::BTreeMap, path::Path};

use serde::{Deserialize, Serialize};

use crate::{ConfigDataType, EmbeddingConfig, EmbeddingConfigCommon, LinearConfig, TransformerConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct TtsMessageProcessorConfig {
    pub prompt_template: String,
    #[serde(default = "default_drop_initial_newline")]
    pub drop_initial_newline: bool,
    #[serde(default = "default_system_role_name")]
    pub system_role_name: String,
    #[serde(default = "default_user_role_name")]
    pub user_role_name: String,
    #[serde(default = "default_assistant_role_name")]
    pub assistant_role_name: String,
    #[serde(default = "default_tts_message_fields")]
    pub default_message_fields: BTreeMap<String, String>,
}

fn default_drop_initial_newline() -> bool {
    true
}

fn default_system_role_name() -> String {
    String::from("system")
}

fn default_user_role_name() -> String {
    String::from("user")
}

fn default_assistant_role_name() -> String {
    String::from("assistant")
}

fn default_tts_message_fields() -> BTreeMap<String, String> {
    BTreeMap::new()
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
    DescriptAudioCodecConfig {
        #[serde(flatten)]
        config: DescriptAudioCodecConfig,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
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

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(deny_unknown_fields)]
pub struct DescriptAudioDacDecoderConfig {
    pub precision: ConfigDataType,
    pub conv_config: DescriptAudioCausalConv1dConfig,
    pub snake_config: DescriptAudioSnake1dConfig,
    pub decoder_block_config: DescriptAudioDacDecoderBlockConfig,
    pub causal: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(deny_unknown_fields)]
pub struct DescriptAudioDacDecoderBlockConfig {
    pub precision: ConfigDataType,
    pub snake_config: DescriptAudioSnake1dConfig,
    pub trans_conv_config: DescriptAudioCausalTransposeConv1dConfig,
    pub res_unit_config: DescriptAudioResidualUnitConfig,
    pub causal: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(deny_unknown_fields)]
pub struct DescriptAudioResidualUnitConfig {
    pub precision: ConfigDataType,
    pub snake_config: DescriptAudioSnake1dConfig,
    pub conv_config: DescriptAudioCausalConv1dConfig,
    pub causal: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(deny_unknown_fields)]
pub struct DescriptAudioCausalConv1dConfig {
    pub precision: ConfigDataType,
    pub has_biases: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(deny_unknown_fields)]
pub struct DescriptAudioCausalTransposeConv1dConfig {
    pub precision: ConfigDataType,
    pub has_biases: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(deny_unknown_fields)]
pub struct DescriptAudioSnake1dConfig {
    pub precision: ConfigDataType,
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
}
