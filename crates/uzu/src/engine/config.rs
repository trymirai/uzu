use std::env;

use serde::{Deserialize, Serialize};

#[bindings::export(Struct, name = "EngineConfig")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Config {
    pub mirai_api_key: Option<String>,
    pub lalamo_path: Option<String>,
    pub huggingface_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub xai_api_key: Option<String>,
    pub baseten_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mirai_api_key: env::var("MIRAI_API_KEY").ok(),
            lalamo_path: env::var("LALAMO_PATH").ok(),
            huggingface_api_key: env::var("HF_TOKEN").ok(),
            openai_api_key: env::var("OPENAI_API_KEY").ok(),
            anthropic_api_key: env::var("ANTHROPIC_API_KEY").ok(),
            gemini_api_key: env::var("GEMINI_API_KEY").ok(),
            xai_api_key: env::var("XAI_API_KEY").ok(),
            baseten_api_key: env::var("BASETEN_API_KEY").ok(),
            openrouter_api_key: env::var("OPENROUTER_API_KEY").ok(),
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

    pub fn with_lalamo_path(
        mut self,
        lalamo_path: String,
    ) -> Self {
        self.lalamo_path = Some(lalamo_path);
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

    pub fn with_anthropic_api_key(
        mut self,
        anthropic_api_key: String,
    ) -> Self {
        self.anthropic_api_key = Some(anthropic_api_key);
        self
    }

    pub fn with_gemini_api_key(
        mut self,
        gemini_api_key: String,
    ) -> Self {
        self.gemini_api_key = Some(gemini_api_key);
        self
    }

    pub fn with_xai_api_key(
        mut self,
        xai_api_key: String,
    ) -> Self {
        self.xai_api_key = Some(xai_api_key);
        self
    }

    pub fn with_baseten_api_key(
        mut self,
        baseten_api_key: String,
    ) -> Self {
        self.baseten_api_key = Some(baseten_api_key);
        self
    }

    pub fn with_openrouter_api_key(
        mut self,
        openrouter_api_key: String,
    ) -> Self {
        self.openrouter_api_key = Some(openrouter_api_key);
        self
    }
}
