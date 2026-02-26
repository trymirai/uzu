use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct TtsMessageProcessorConfig {
    pub prompt_template: String,
    #[serde(default = "default_drop_initial_newline")]
    pub drop_initial_newline: bool,
}

fn default_drop_initial_newline() -> bool {
    true
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct TtsModelConfig {
    pub tts_config: serde_json::Value,
    pub message_processor_config: TtsMessageProcessorConfig,
}

#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
impl TtsModelConfig {
    pub fn create_audio_codec_runtime(&self) -> crate::audio::AudioResult<crate::audio::NanoCodecFsqRuntime> {
        crate::audio::NanoCodecFsqRuntime::from_tts_config_value(&self.tts_config)
    }

    pub fn create_audio_codec_runtime_with_model_path(
        &self,
        model_path: &Path,
    ) -> crate::audio::AudioResult<crate::audio::NanoCodecFsqRuntime> {
        crate::audio::NanoCodecFsqRuntime::from_tts_config_value_and_model_path(&self.tts_config, model_path)
    }
}
