use proc_macros::uzu_config;

use crate::config::{DecoderConfig, GenerationConfig, MessageProcessorConfig};

#[uzu_config]
pub struct LanguageModelConfig {
    pub model_config: DecoderConfig,
    pub message_processor_config: MessageProcessorConfig,
    pub generation_config: GenerationConfig,
}
