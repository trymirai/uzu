use shoji::types::{model::EncodingConfig, session::chat::ChatModelCapabilities};

use crate::chat::{Error, hanashi::config::hanashi_config_capabilities, harmony::harmony_config_capabilities};

#[allow(dead_code)]
pub fn encoding_config_capabilities(config: &EncodingConfig) -> Result<ChatModelCapabilities, Error> {
    match config {
        EncodingConfig::Hanashi(config) => hanashi_config_capabilities(config).map_err(Error::from),
        EncodingConfig::Harmony(config) => Ok(harmony_config_capabilities(config)),
    }
}
