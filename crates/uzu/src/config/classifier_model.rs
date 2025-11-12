use serde::{Deserialize, Serialize};

use crate::{ClassifierConfig, MessageProcessorConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ClassifierModelConfig {
    pub classifier_config: ClassifierConfig,
    pub message_processor_config: MessageProcessorConfig,
}
