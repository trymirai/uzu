use serde::{Deserialize, Serialize};

use super::DecoderConfig;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ModelConfig {
    pub decoder_config: DecoderConfig,
}
