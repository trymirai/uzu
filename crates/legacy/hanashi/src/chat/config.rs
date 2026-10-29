use serde::{Deserialize, Serialize};
use shoji::types::{
    basic::{ReasoningEffort, Value},
    session::chat::ChatModelCapabilities,
};

#[cfg(not(target_family = "wasm"))]
use crate::chat::harmony::HarmonyConfig;
use crate::chat::{Error, hanashi::config::HanashiConfig};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EncodingConfig {
    Hanashi {
        #[serde(flatten)]
        config: HanashiConfig,
    },
    #[cfg(not(target_family = "wasm"))]
    Harmony {
        #[serde(flatten)]
        config: HarmonyConfig,
    },
}

impl EncodingConfig {
    pub fn default_reasoning_effort(&self) -> Result<Option<ReasoningEffort>, Error> {
        match self {
            EncodingConfig::Hanashi {
                config,
            } => config.default_reasoning_effort().map_err(Error::from),
            #[cfg(not(target_family = "wasm"))]
            EncodingConfig::Harmony {
                config,
            } => Ok(config.default_reasoning_effort()),
        }
    }

    pub fn select(encodings: &[Value]) -> Option<Value> {
        let parse = |value: &Value| serde_json::from_str::<EncodingConfig>(&value.json).ok();
        #[cfg(not(target_family = "wasm"))]
        if let Some(value) = encodings.iter().find(|value| matches!(parse(value), Some(EncodingConfig::Harmony { .. })))
        {
            return Some(value.clone());
        }
        encodings.iter().find(|value| parse(value).is_some()).cloned()
    }

    pub fn capabilities(&self) -> Result<ChatModelCapabilities, Error> {
        match self {
            EncodingConfig::Hanashi {
                config,
            } => config.capabilities().map_err(Error::from),
            #[cfg(not(target_family = "wasm"))]
            EncodingConfig::Harmony {
                config,
            } => Ok(config.capabilities()),
        }
    }
}
