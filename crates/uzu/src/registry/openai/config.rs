use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Config {
    pub identifier: String,
    pub name: String,
    pub api_endpoint: String,
    pub api_key: Option<String>,
    pub model_filter_pattern: Option<String>,
    pub headers: Option<HashMap<String, String>>,
}

impl Config {
    pub fn new(
        identifier: String,
        name: String,
        api_endpoint: String,
        api_key: Option<String>,
        model_filter_pattern: Option<String>,
        headers: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            identifier,
            name,
            api_endpoint,
            api_key,
            model_filter_pattern,
            headers,
        }
    }

    pub fn openai(api_key: String) -> Self {
        Self::new(
            "openai".to_string(),
            "OpenAI".to_string(),
            "https://api.openai.com/v1".to_string(),
            Some(api_key),
            Some(r"^(?:gpt-\d[\d.]*[a-z]?|o\d+)(?:-(?:mini|nano|pro|turbo|codex(?:-mini|-max)?))*$".to_string()),
            None,
        )
    }

    pub fn ollama() -> Self {
        Self::new("ollama".to_string(), "Ollama".to_string(), "http://localhost:11434/v1".to_string(), None, None, None)
    }

    pub fn baseten(api_key: String) -> Self {
        Self::new(
            "baseten".to_string(),
            "Baseten".to_string(),
            "https://inference.baseten.co/v1".to_string(),
            Some(api_key),
            None,
            None,
        )
    }

    pub fn anthropic(api_key: String) -> Self {
        Self::new(
            "anthropic".to_string(),
            "Anthropic".to_string(),
            "https://api.anthropic.com/v1/".to_string(),
            Some(api_key.clone()),
            Some(r"^claude-(?:opus|sonnet|haiku)-\d{1,2}(?:-\d{1,2})*$".to_string()),
            Some(HashMap::from([
                ("X-Api-Key".to_string(), api_key.clone()),
                ("anthropic-version".to_string(), "2023-06-01".to_string()),
            ])),
        )
    }

    pub fn open_router(api_key: String) -> Self {
        Self::new(
            "openrouter".to_string(),
            "OpenRouter".to_string(),
            "https://openrouter.ai/api/v1".to_string(),
            Some(api_key.clone()),
            None,
            None,
        )
    }
}
