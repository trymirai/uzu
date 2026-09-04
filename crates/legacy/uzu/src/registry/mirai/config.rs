use std::path::PathBuf;

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
    pub device: Device,
    pub backends: Vec<Backend>,
    pub include_traces: bool,
    pub cache_path: PathBuf,
}

impl Config {
    pub fn new(
        api_key: Option<String>,
        device: Device,
        backends: Vec<Backend>,
        include_traces: bool,
        cache_path: PathBuf,
    ) -> Self {
        Self {
            api_key,
            device,
            backends,
            include_traces,
            cache_path,
        }
    }
}
