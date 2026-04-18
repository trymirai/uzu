use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Config {
    pub identifier: String,
    pub name: String,
    pub api_endpoint: String,
    pub api_key: Option<String>,
    pub model_filter_pattern: Option<String>,
}

impl Config {
    pub fn new(
        identifier: String,
        name: String,
        api_endpoint: String,
        api_key: Option<String>,
        model_filter_pattern: Option<String>,
    ) -> Self {
        Self {
            identifier,
            name,
            api_endpoint,
            api_key,
            model_filter_pattern,
        }
    }

    pub fn openai(api_key: String) -> Self {
        Self::new(
            "openai".to_string(),
            "OpenAI".to_string(),
            "https://api.openai.com/v1".to_string(),
            Some(api_key),
            Some(r"^(?:gpt-\d[\d.]*[a-z]?|o\d+)(?:-(?:mini|nano|pro|turbo|codex(?:-mini|-max)?))*$".to_string()),
        )
    }

    pub fn ollama() -> Self {
        Self::new("ollama".to_string(), "Ollama".to_string(), "http://localhost:11434/v1".to_string(), None, None)
    }

    pub fn baseten(api_key: String) -> Self {
        Self::new(
            "baseten".to_string(),
            "Baseten".to_string(),
            "https://inference.baseten.co/v1".to_string(),
            Some(api_key),
            None,
        )
    }
}
