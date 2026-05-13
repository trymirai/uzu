use serde::{Deserialize, Serialize};

use crate::config::{DecoderConfig, GenerationConfig, MessageProcessorConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct LanguageModelConfig {
    pub model_config: DecoderConfig,
    pub message_processor_config: MessageProcessorConfig,
    pub generation_config: GenerationConfig,
}
