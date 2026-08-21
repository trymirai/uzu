use std::collections::HashMap;

use backend_remote::openai::{ApiType as BackendApiType, Config as BackendConfig};
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
    pub name: String,
    pub api_endpoint: String,
    pub api_key: Option<String>,
    pub api_type: ApiType,
    pub model_filter_pattern: Option<String>,
    pub headers: Option<HashMap<String, String>>,
}

impl Config {
    pub fn new(
        identifier: String,
        name: String,
        api_endpoint: String,
        api_key: Option<String>,
        api_type: ApiType,
        model_filter_pattern: Option<String>,
        headers: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            identifier,
            name,
            api_endpoint,
            api_key,
            api_type,
            model_filter_pattern,
            headers,
        }
    }

    pub fn ollama() -> Self {
        Self::new(
            "ollama".to_string(),
            "Ollama".to_string(),
            "http://localhost:11434/v1".to_string(),
            None,
            ApiType::Responses,
            None,
            None,
        )
    }

    pub fn lmstudio() -> Self {
        Self::new(
            "lmstudio".to_string(),
            "LM Studio".to_string(),
            "http://localhost:1234/v1".to_string(),
            None,
            ApiType::Responses,
            Some(r"^(?!.*embedding).+$".to_string()),
            None,
        )
    }

    pub fn openai(api_key: String) -> Self {
        Self::new(
            "openai".to_string(),
            "OpenAI".to_string(),
            "https://api.openai.com/v1".to_string(),
            Some(api_key),
            ApiType::Responses,
            Some(r"^(?:gpt-\d[\d.]*[a-z]?|o\d+)(?:-(?:mini|nano|pro|turbo|codex(?:-mini|-max)?))*$".to_string()),
            None,
        )
    }

    pub fn anthropic(api_key: String) -> Self {
        Self::new(
            "anthropic".to_string(),
            "Anthropic".to_string(),
            "https://api.anthropic.com/v1".to_string(),
            Some(api_key.clone()),
            ApiType::Completions,
            Some(r"^claude-(?:opus|sonnet|haiku)-\d{1,2}(?:-\d{1,2})*$".to_string()),
            Some(HashMap::from([
                ("X-Api-Key".to_string(), api_key.clone()),
                ("anthropic-version".to_string(), "2023-06-01".to_string()),
            ])),
        )
    }

    pub fn gemini(api_key: String) -> Self {
        Self::new(
            "gemini".to_string(),
            "Gemini".to_string(),
            "https://generativelanguage.googleapis.com/v1beta/openai".to_string(),
            Some(api_key),
            ApiType::Completions,
            Some(r"^models/(?:gemini|gemma)-(?!.*(?:image|audio|tts|live|embedding|computer-use|robotics|preview|latest|-\d{3}$|-\d{2}-\d{4}$)).+$".to_string()),
            None,
        )
    }

    pub fn xai(api_key: String) -> Self {
        Self::new(
            "xai".to_string(),
            "xAI".to_string(),
            "https://api.x.ai/v1".to_string(),
            Some(api_key),
            ApiType::Responses,
            Some(r"^grok-(?!.*(?:imagine|-\d{4})).+$".to_string()),
            None,
        )
    }

    pub fn baseten(api_key: String) -> Self {
        Self::new(
            "baseten".to_string(),
            "Baseten".to_string(),
            "https://inference.baseten.co/v1".to_string(),
            Some(api_key),
            ApiType::Completions,
            None,
            None,
        )
    }

    pub fn openrouter(api_key: String) -> Self {
        Self::new(
            "openrouter".to_string(),
            "OpenRouter".to_string(),
            "https://openrouter.ai/api/v1".to_string(),
            Some(api_key.clone()),
            ApiType::Responses,
            None,
            None,
        )
    }
}

impl From<Config> for BackendConfig {
    fn from(config: Config) -> Self {
        Self {
            identifier: config.identifier,
            api_endpoint: config.api_endpoint,
            api_key: config.api_key,
            api_type: config.api_type.into(),
            headers: config.headers,
        }
    }
}

impl From<ApiType> for BackendApiType {
    fn from(value: ApiType) -> Self {
        match value {
            ApiType::Completions => BackendApiType::Completions,
            ApiType::Responses => BackendApiType::Responses,
        }
    }
}
