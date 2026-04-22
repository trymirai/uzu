use std::env;

use serde::{Deserialize, Serialize};

#[bindings::export(Class)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EngineConfig {
    pub mirai_api_key: Option<String>,
    pub lalamo_path: Option<String>,
    pub huggingface_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub xai_api_key: Option<String>,
    pub baseten_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
    pub allow_ollama_usage: bool,
    pub allow_lmstudio_usage: bool,
}

#[bindings::export(Implementation)]
impl EngineConfig {
    #[bindings::export(Constructor)]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for EngineConfig {
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
            allow_ollama_usage: true,
            allow_lmstudio_usage: true,
        }
    }
}

#[bindings::export(Implementation)]
impl EngineConfig {
    #[bindings::export(Method)]
    pub fn with_mirai_api_key(
        &self,
        mirai_api_key: String,
    ) -> Self {
        Self {
            mirai_api_key: Some(mirai_api_key),
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_lalamo_path(
        &self,
        lalamo_path: String,
    ) -> Self {
        Self {
            lalamo_path: Some(lalamo_path),
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_huggingface_api_key(
        &self,
        huggingface_api_key: String,
    ) -> Self {
        Self {
            huggingface_api_key: Some(huggingface_api_key),
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_openai_api_key(
        &self,
        openai_api_key: String,
    ) -> Self {
        Self {
            openai_api_key: Some(openai_api_key),
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_anthropic_api_key(
        &self,
        anthropic_api_key: String,
    ) -> Self {
        Self {
            anthropic_api_key: Some(anthropic_api_key),
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_gemini_api_key(
        &self,
        gemini_api_key: String,
    ) -> Self {
        Self {
            gemini_api_key: Some(gemini_api_key),
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_xai_api_key(
        &self,
        xai_api_key: String,
    ) -> Self {
        Self {
            xai_api_key: Some(xai_api_key),
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_baseten_api_key(
        &self,
        baseten_api_key: String,
    ) -> Self {
        Self {
            baseten_api_key: Some(baseten_api_key),
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_openrouter_api_key(
        &self,
        openrouter_api_key: String,
    ) -> Self {
        Self {
            openrouter_api_key: Some(openrouter_api_key),
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_allow_ollama_usage(
        &self,
        allow_ollama_usage: bool,
    ) -> Self {
        Self {
            allow_ollama_usage,
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_allow_lmstudio_usage(
        &self,
        allow_lmstudio_usage: bool,
    ) -> Self {
        Self {
            allow_lmstudio_usage,
            ..self.clone()
        }
    }
}
