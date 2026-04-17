use std::env;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Config {
    pub mirai_api_key: Option<String>,
    pub huggingface_api_key: Option<String>,
    pub openai_api_key: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mirai_api_key: env::var("MIRAI_API_KEY").ok(),
            huggingface_api_key: env::var("HF_TOKEN").ok(),
            openai_api_key: env::var("OPENAI_API_KEY").ok(),
        }
    }
}

impl Config {
    pub fn with_mirai_api_key(
        mut self,
        mirai_api_key: String,
    ) -> Self {
        self.mirai_api_key = Some(mirai_api_key);
        self
    }

    pub fn with_huggingface_api_key(
        mut self,
        huggingface_api_key: String,
    ) -> Self {
        self.huggingface_api_key = Some(huggingface_api_key);
        self
    }

    pub fn with_openai_api_key(
        mut self,
        openai_api_key: String,
    ) -> Self {
        self.openai_api_key = Some(openai_api_key);
        self
    }
}
