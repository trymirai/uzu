use std::env;

use serde::{Deserialize, Serialize};

use crate::settings::{SettingType, Settings, SettingsError};

const KEY_MIRAI_API_KEY: &str = "MIRAI_API_KEY";
const KEY_LALAMO_PATH: &str = "LALAMO_PATH";
const KEY_HF_TOKEN: &str = "HF_TOKEN";
const KEY_OPENAI_API_KEY: &str = "OPENAI_API_KEY";
const KEY_ANTHROPIC_API_KEY: &str = "ANTHROPIC_API_KEY";
const KEY_GEMINI_API_KEY: &str = "GEMINI_API_KEY";
const KEY_XAI_API_KEY: &str = "XAI_API_KEY";
const KEY_BASETEN_API_KEY: &str = "BASETEN_API_KEY";
const KEY_OPENROUTER_API_KEY: &str = "OPENROUTER_API_KEY";

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EngineConfig {
    pub application_identifier: Option<String>,
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

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            application_identifier: None,
            mirai_api_key: env::var(KEY_MIRAI_API_KEY).ok(),
            lalamo_path: env::var(KEY_LALAMO_PATH).ok(),
            huggingface_api_key: env::var(KEY_HF_TOKEN).ok(),
            openai_api_key: env::var(KEY_OPENAI_API_KEY).ok(),
            anthropic_api_key: env::var(KEY_ANTHROPIC_API_KEY).ok(),
            gemini_api_key: env::var(KEY_GEMINI_API_KEY).ok(),
            xai_api_key: env::var(KEY_XAI_API_KEY).ok(),
            baseten_api_key: env::var(KEY_BASETEN_API_KEY).ok(),
            openrouter_api_key: env::var(KEY_OPENROUTER_API_KEY).ok(),
            allow_ollama_usage: true,
            allow_lmstudio_usage: true,
        }
    }
}

impl EngineConfig {
    pub fn synchronize_with_settings(
        &mut self,
        settings: &Settings,
    ) -> Result<(), SettingsError> {
        macro_rules! synchronize_field {
            ($setting_type:path, $field:ident, $key:expr) => {
                if let Some(value) = &self.$field {
                    settings.save($setting_type, $key, Some(value.clone()))?;
                } else if let Some(value) = settings.load($setting_type, $key)? {
                    self.$field = Some(value);
                }
            };
        }

        synchronize_field!(SettingType::Secret, mirai_api_key, KEY_MIRAI_API_KEY);
        synchronize_field!(SettingType::Secret, lalamo_path, KEY_LALAMO_PATH);
        synchronize_field!(SettingType::Secret, huggingface_api_key, KEY_HF_TOKEN);
        synchronize_field!(SettingType::Secret, openai_api_key, KEY_OPENAI_API_KEY);
        synchronize_field!(SettingType::Secret, anthropic_api_key, KEY_ANTHROPIC_API_KEY);
        synchronize_field!(SettingType::Secret, gemini_api_key, KEY_GEMINI_API_KEY);
        synchronize_field!(SettingType::Secret, xai_api_key, KEY_XAI_API_KEY);
        synchronize_field!(SettingType::Secret, baseten_api_key, KEY_BASETEN_API_KEY);
        synchronize_field!(SettingType::Secret, openrouter_api_key, KEY_OPENROUTER_API_KEY);

        Ok(())
    }
}

#[bindings::export(Implementation)]
impl EngineConfig {
    #[bindings::export(Method(Factory))]
    pub fn create() -> Self {
        Self::default()
    }
}

#[bindings::export(Implementation)]
impl EngineConfig {
    #[bindings::export(Method)]
    pub fn with_application_identifier(
        &self,
        application_identifier: String,
    ) -> Self {
        Self {
            application_identifier: Some(application_identifier),
            ..self.clone()
        }
    }

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
