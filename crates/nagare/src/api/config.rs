use std::time::Duration;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

const MIRAI_BASE_URL: &str = "https://sdk.trymirai.com/api/v1";

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BaseUrl {
    Mirai,
    Custom(String),
}

impl Default for BaseUrl {
    fn default() -> Self {
        Self::Mirai
    }
}

impl BaseUrl {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Mirai => MIRAI_BASE_URL,
            Self::Custom(base_url) => base_url,
        }
    }
}

pub struct Config {
    pub base_url: BaseUrl,
    pub timeout: Duration,
    pub headers: IndexMap<String, String>,
}

impl Config {
    pub fn new(
        base_url: BaseUrl,
        timeout: Duration,
        headers: IndexMap<String, String>,
    ) -> Self {
        Self {
            base_url,
            timeout,
            headers,
        }
    }
}
