use serde::{Deserialize, Serialize};
use shoji::types::encoding::Capabilities;

use crate::chat::{Error, hanashi::Config as HanashiConfig, harmony::Config as HarmonyConfig};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Config {
    Hanashi(HanashiConfig),
    Harmony(HarmonyConfig),
}

impl Config {
    pub fn capabilities(&self) -> Result<Capabilities, Error> {
        match self {
            Config::Hanashi(config) => config.capabilities().map_err(Error::from),
            Config::Harmony(config) => Ok(config.capabilities()),
        }
    }
}
