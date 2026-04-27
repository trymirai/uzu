use nagare::api::BaseUrl;
use serde::{Deserialize, Serialize};

use crate::device::Device;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Backend {
    pub identifier: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Config {
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: BaseUrl,
    pub device: Device,
    pub backends: Vec<Backend>,
    pub include_traces: bool,
}

impl Config {
    pub fn new(
        api_key: Option<String>,
        base_url: BaseUrl,
        device: Device,
        backends: Vec<Backend>,
        include_traces: bool,
    ) -> Self {
        Self {
            api_key,
            base_url,
            device,
            backends,
            include_traces,
        }
    }
}
