use serde::{Deserialize, Serialize};
use shoji::types::{basic::Value, session::chat::ChatModelCapabilities};

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

impl EncodingConfig {
    pub fn select(encodings: &[Value]) -> Option<Value> {
        let parse = |value: &Value| serde_json::from_str::<EncodingConfig>(&value.json).ok();
        encodings
            .iter()
            .find(|value| matches!(parse(value), Some(EncodingConfig::Harmony { .. })))
            .or_else(|| encodings.iter().find(|value| parse(value).is_some()))
            .cloned()
    }

    pub fn capabilities(&self) -> Result<ChatModelCapabilities, Error> {
        match self {
            EncodingConfig::Hanashi {
                config,
            } => config.capabilities().map_err(Error::from),
            EncodingConfig::Harmony {
                config,
            } => Ok(config.capabilities()),
        }
    }
}
