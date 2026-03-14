#![cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]

use std::{path::Path, rc::Rc};

use super::{AudioResult, NanoCodecFsqRuntime, NanoCodecFsqRuntimeOptions};
use crate::config::TtsConfig;

#[derive(Clone)]
pub struct AudioGenerationContext {
    runtime: Rc<NanoCodecFsqRuntime>,
    codec_cardinality: usize,
    semantic_codec_cardinality: Option<usize>,
    num_codebooks: usize,
    sample_rate: u32,
}

impl AudioGenerationContext {
    pub fn from_tts_config_and_model_path(
        tts_config: &TtsConfig,
        model_path: &Path,
    ) -> AudioResult<Self> {
        Self::with_runtime(NanoCodecFsqRuntime::from_tts_config_and_model_path(tts_config, model_path)?)
    }

    pub fn from_tts_config_and_model_path_with_options(
        tts_config: &TtsConfig,
        model_path: &Path,
        options: NanoCodecFsqRuntimeOptions,
    ) -> AudioResult<Self> {
        Self::with_runtime(NanoCodecFsqRuntime::from_tts_config_and_model_path_with_options(
            tts_config, model_path, options,
        )?)
    }

    fn with_runtime(runtime: NanoCodecFsqRuntime) -> AudioResult<Self> {
        let codec_cardinality = usize::try_from(runtime.config().codec_cardinality())
            .map_err(|_| super::AudioError::Runtime("audio codec cardinality exceeds usize".to_string()))?;
        Ok(Self {
            codec_cardinality,
            semantic_codec_cardinality: runtime.config().semantic_codec_cardinality(),
            num_codebooks: runtime.config().num_groups(),
            sample_rate: runtime.config().sample_rate(),
            runtime: Rc::new(runtime),
        })
    }

    pub fn runtime(&self) -> &NanoCodecFsqRuntime {
        self.runtime.as_ref()
    }

    pub fn codec_cardinality(&self) -> usize {
        self.codec_cardinality
    }

    pub fn semantic_codec_cardinality(&self) -> Option<usize> {
        self.semantic_codec_cardinality
    }

    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
