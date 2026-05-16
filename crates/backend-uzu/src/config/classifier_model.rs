use proc_macros::uzu_config;

use crate::config::{ClassifierConfig, MessageProcessorConfig};

#[uzu_config]
pub struct ClassifierModelConfig {
    pub model_config: ClassifierConfig,
    pub message_processor_config: MessageProcessorConfig,
}
