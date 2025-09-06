use serde::{Deserialize, Serialize};

use super::{DecoderConfig, GenerationConfig, MessageProcessorConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct LanguageModelConfig {
    pub decoder_config: DecoderConfig,
    pub message_processor_config: MessageProcessorConfig,
    pub generation_config: GenerationConfig,
}
