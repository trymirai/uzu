use uzu::{
    engine::{
        Engine,
        config::{
            KEY_ANTHROPIC_API_KEY, KEY_BASETEN_API_KEY, KEY_GEMINI_API_KEY, KEY_OPENAI_API_KEY, KEY_OPENROUTER_API_KEY,
            KEY_XAI_API_KEY,
        },
    },
    registry::openai::Config as OpenAiConfig,
    settings::SettingKind,
};

pub const APPLICATION_ID: &str = "Mirai";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CloudProvider {
    pub id: &'static str,
    pub label: &'static str,
    pub settings_key: &'static str,
}

pub const PROVIDERS: &[CloudProvider] = &[
    CloudProvider {
        id: "openai",
        label: "OpenAI",
        settings_key: KEY_OPENAI_API_KEY,
    },
    CloudProvider {
        id: "anthropic",
        label: "Anthropic",
        settings_key: KEY_ANTHROPIC_API_KEY,
    },
    CloudProvider {
        id: "gemini",
        label: "Google Gemini",
        settings_key: KEY_GEMINI_API_KEY,
    },
    CloudProvider {
        id: "xai",
        label: "xAI",
        settings_key: KEY_XAI_API_KEY,
    },
    CloudProvider {
        id: "baseten",
        label: "Baseten",
        settings_key: KEY_BASETEN_API_KEY,
    },
    CloudProvider {
        id: "openrouter",
        label: "OpenRouter",
        settings_key: KEY_OPENROUTER_API_KEY,
    },
];

pub fn provider_by_id(id: &str) -> Option<&'static CloudProvider> {
    PROVIDERS.iter().find(|p| p.id == id)
}

fn openai_config(
    provider_id: &str,
    api_key: String,
) -> Option<OpenAiConfig> {
    Some(match provider_id {
        "openai" => OpenAiConfig::openai(api_key),
        "anthropic" => OpenAiConfig::anthropic(api_key),
        "gemini" => OpenAiConfig::gemini(api_key),
        "xai" => OpenAiConfig::xai(api_key),
        "baseten" => OpenAiConfig::baseten(api_key),
        "openrouter" => OpenAiConfig::openrouter(api_key),
        _ => return None,
    })
}

pub async fn load_key(
    engine: &Engine,
    settings_key: &str,
) -> Option<String> {
    let settings = engine.settings().await.ok()?;
    settings.load(SettingKind::Secret, settings_key.to_string()).ok().flatten().filter(|v| !v.trim().is_empty())
}

pub async fn configured_providers(engine: &Engine) -> Vec<(&'static CloudProvider, bool)> {
    let mut out = Vec::new();
    for provider in PROVIDERS {
        let configured = load_key(engine, provider.settings_key).await.is_some();
        out.push((provider, configured));
    }
    out
}

pub async fn set_provider_key(
    engine: &Engine,
    provider_id: &str,
    api_key: Option<String>,
) -> Result<(), String> {
    let provider = provider_by_id(provider_id).ok_or("unknown provider")?;
    let settings = engine.settings().await.map_err(|e| format!("settings unavailable: {e}"))?;

    let trimmed = api_key.as_deref().map(str::trim).filter(|s| !s.is_empty());

    if let Some(key) = trimmed {
        let config = openai_config(provider.id, key.to_string()).ok_or("unknown provider")?;

        engine.connect_openai(config).await.map_err(|e| e.to_string())?;
    } else {
        let _ = engine.remove_registry(provider.id.to_string()).await;
        engine.remove_backend(provider.id.to_string()).await;
    }
    settings
        .save(SettingKind::Secret, provider.settings_key.to_string(), trimmed.map(str::to_string))
        .map_err(|e| e.to_string())?;
    Ok(())
}
