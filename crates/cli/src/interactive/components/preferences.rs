use std::{collections::HashMap, error::Error, fs, io, path::Path};

use serde::{Deserialize, Serialize};

use crate::{
    common::model_capabilities::ThinkingPreference,
    interactive::{components::Theme, sampling::SamplingPreferences},
};

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Preferences {
    pub theme: Theme,
    pub thinking: ThinkingPreference,
    pub sampling: SamplingPreferences,
    pub selected_model_id: Option<String>,
}

impl Preferences {
    pub(super) fn load_or_migrate(
        config_path: &Path,
        legacy_path: &Path,
    ) -> Result<Self, Box<dyn Error>> {
        if !legacy_path.exists() {
            return if config_path.exists() {
                Self::load(config_path)
            } else {
                Ok(Self::default())
            };
        }

        let mut preferences = if config_path.exists() {
            Self::load(config_path)?
        } else {
            Self::default()
        };

        let mut legacy: HashMap<String, HashMap<String, String>> = toml::from_str(&fs::read_to_string(legacy_path)?)?;
        let settings = legacy.remove("settings").unwrap_or_default();
        if let Some(raw) = settings.get("cli_preferences")
            && let Ok(legacy_preferences) = serde_json::from_str::<Self>(raw)
        {
            preferences.thinking = legacy_preferences.thinking;
            preferences.sampling = legacy_preferences.sampling;
        }
        if let Some(raw) = settings.get("app") {
            let app_settings: Self = serde_json::from_str(raw)?;
            preferences.selected_model_id = app_settings.selected_model_id;
        }
        if let Some(name) = settings.get("theme") {
            preferences.theme = Theme::from_name(name)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("unknown legacy theme: {name}")))?;
        }

        preferences.store(config_path)?;
        Ok(preferences)
    }

    pub(super) fn store(
        &self,
        path: &Path,
    ) -> Result<(), Box<dyn Error>> {
        let parent = path
            .parent()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "settings file has no parent directory"))?;
        fs::create_dir_all(parent)?;
        fs::write(path, toml::to_string(self)?)?;
        Ok(())
    }

    fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let contents = fs::read_to_string(path)?;
        toml::from_str(&contents).map_err(Into::into)
    }
}
