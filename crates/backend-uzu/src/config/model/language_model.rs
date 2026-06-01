use proc_macros::uzu_config;

use crate::config::{decoder::DecoderConfig, model::generation::GenerationConfig};

#[uzu_config(super::ModelConfig)]
pub struct LanguageModelConfig {
    pub decoder_config: DecoderConfig,
    pub generation_config: GenerationConfig,
}
