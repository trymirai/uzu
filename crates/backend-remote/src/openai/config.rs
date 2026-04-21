use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiType {
    Completions,
    Responses,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Config {
    pub identifier: String,
    pub api_endpoint: String,
    pub api_key: Option<String>,
    pub api_type: ApiType,
    pub headers: Option<HashMap<String, String>>,
}
