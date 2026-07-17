use serde::{Deserialize, Serialize};
use shoji::types::session::chat::ChatModelCapabilities;

use crate::chat::{Error, hanashi::config::HanashiConfig, harmony::HarmonyConfig};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EncodingConfig {
    Hanashi {
        #[serde(flatten)]
        config: HanashiConfig,
    },
    Harmony {
        #[serde(flatten)]
        config: HarmonyConfig,
    },
}

#[allow(dead_code)]
pub fn encoding_config_capabilities(config: &EncodingConfig) -> Result<ChatModelCapabilities, Error> {
    match config {
        EncodingConfig::Hanashi {
            config,
        } => config.capabilities().map_err(Error::from),
        EncodingConfig::Harmony {
            config,
        } => Ok(config.capabilities()),
    }
}
