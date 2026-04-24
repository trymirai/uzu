use std::env;

use serde::{Deserialize, Serialize};

use crate::keyring::Keyring;

const KEY_MIRAI_API_KEY: &str = "MIRAI_API_KEY";
const KEY_LALAMO_PATH: &str = "LALAMO_PATH";
const KEY_HF_TOKEN: &str = "HF_TOKEN";
const KEY_OPENAI_API_KEY: &str = "OPENAI_API_KEY";
const KEY_ANTHROPIC_API_KEY: &str = "ANTHROPIC_API_KEY";
const KEY_GEMINI_API_KEY: &str = "GEMINI_API_KEY";
const KEY_XAI_API_KEY: &str = "XAI_API_KEY";
const KEY_BASETEN_API_KEY: &str = "BASETEN_API_KEY";
const KEY_OPENROUTER_API_KEY: &str = "OPENROUTER_API_KEY";

#[bindings::export(ClassCloneable)]
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
    #[bindings::export(Factory)]
    pub fn create() -> Self {
        Self::default()
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        let keyring = process_env_to_keyring();
        Self {
            mirai_api_key: retrieve_keyring_or_env_value(&keyring, KEY_MIRAI_API_KEY),
            lalamo_path: retrieve_keyring_or_env_value(&keyring, KEY_LALAMO_PATH),
            huggingface_api_key: retrieve_keyring_or_env_value(&keyring, KEY_HF_TOKEN),
            openai_api_key: retrieve_keyring_or_env_value(&keyring, KEY_OPENAI_API_KEY),
            anthropic_api_key: retrieve_keyring_or_env_value(&keyring, KEY_ANTHROPIC_API_KEY),
            gemini_api_key: retrieve_keyring_or_env_value(&keyring, KEY_GEMINI_API_KEY),
            xai_api_key: retrieve_keyring_or_env_value(&keyring, KEY_XAI_API_KEY),
            baseten_api_key: retrieve_keyring_or_env_value(&keyring, KEY_BASETEN_API_KEY),
            openrouter_api_key: retrieve_keyring_or_env_value(&keyring, KEY_OPENROUTER_API_KEY),
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
        store_keyring_pair(KEY_MIRAI_API_KEY, &mirai_api_key);
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
        store_keyring_pair(KEY_LALAMO_PATH, &lalamo_path);
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
        store_keyring_pair(KEY_HF_TOKEN, &huggingface_api_key);
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
        store_keyring_pair(KEY_OPENAI_API_KEY, &openai_api_key);
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
        store_keyring_pair(KEY_ANTHROPIC_API_KEY, &anthropic_api_key);
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
        store_keyring_pair(KEY_GEMINI_API_KEY, &gemini_api_key);
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
        store_keyring_pair(KEY_XAI_API_KEY, &xai_api_key);
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
        store_keyring_pair(KEY_BASETEN_API_KEY, &baseten_api_key);
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
        store_keyring_pair(KEY_OPENROUTER_API_KEY, &openrouter_api_key);
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

fn process_env_to_keyring() -> Option<Keyring> {
    let keys = vec![
        KEY_MIRAI_API_KEY,
        KEY_LALAMO_PATH,
        KEY_HF_TOKEN,
        KEY_OPENAI_API_KEY,
        KEY_ANTHROPIC_API_KEY,
        KEY_GEMINI_API_KEY,
        KEY_XAI_API_KEY,
        KEY_BASETEN_API_KEY,
        KEY_OPENROUTER_API_KEY,
    ];

    let keyring = Keyring::new().ok()?;
    for key in keys {
        if let Ok(env_value) = env::var(key) {
            keyring.store(key.to_string(), env_value).ok();
        }
    }
    Some(keyring)
}

fn retrieve_keyring_or_env_value(
    keyring: &Option<Keyring>,
    key: &str,
) -> Option<String> {
    keyring.as_ref().and_then(|keyring| keyring.retrieve(key.to_string())).or_else(|| env::var(key).ok())
}

fn store_keyring_pair(
    key: &str,
    value: &str,
) -> bool {
    if let Ok(keyring) = Keyring::new() {
        keyring.store(key.to_string(), value.to_string()).is_ok()
    } else {
        false
    }
}
