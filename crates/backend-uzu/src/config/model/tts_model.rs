use proc_macros::uzu_config;

use crate::config::tts::TTSConfig;

#[uzu_config(super::ModelConfig)]
pub struct TTSModelConfig {
    pub tts_config: TTSConfig,
}
