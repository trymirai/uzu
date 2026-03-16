#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
use crate::audio::NanoCodecFsqRuntimeOptions;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextSamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
}

impl Default for TextSamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8008,
            top_p: 0.8008,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextDecoderFollowupStrategy {
    SequentialExact,
    AsyncChain,
}

impl Default for TextDecoderFollowupStrategy {
    fn default() -> Self {
        Self::AsyncChain
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextDecoderRuntimeConfig {
    pub sampling: TextSamplingConfig,
    pub min_frames_before_im_end: usize,
    pub max_new_tokens_override: Option<usize>,
    pub force_semantic_sampling_mask: Option<bool>,
    pub prefill_step_size: usize,
    pub followup_strategy: TextDecoderFollowupStrategy,
}

impl Default for TextDecoderRuntimeConfig {
    fn default() -> Self {
        Self {
            sampling: TextSamplingConfig::default(),
            min_frames_before_im_end: 8,
            max_new_tokens_override: None,
            force_semantic_sampling_mask: None,
            prefill_step_size: 128,
            followup_strategy: TextDecoderFollowupStrategy::default(),
        }
    }
}

#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TtsSessionOptions {
    pub text_decoder: TextDecoderRuntimeConfig,
    pub audio_runtime: NanoCodecFsqRuntimeOptions,
}
