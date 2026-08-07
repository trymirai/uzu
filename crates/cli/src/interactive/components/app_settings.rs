use serde::{Deserialize, Serialize};
use uzu::settings::{SettingKind, Settings, SettingsError};

const SETTINGS_APP: &str = "app";

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct AppSettings {
    pub selected_model_id: Option<String>,
}

impl AppSettings {
    pub fn load(settings: &Settings) -> Result<Self, SettingsError> {
        let Some(raw) = settings.load(SettingKind::Config, SETTINGS_APP.to_string())? else {
            return Ok(Self::default());
        };
        Ok(serde_json::from_str(&raw).unwrap_or_default())
    }

    pub fn save(
        &self,
        settings: &Settings,
    ) -> Result<(), SettingsError> {
        let raw = serde_json::to_string(self).map_err(|error| SettingsError::BackendError {
            message: error.to_string(),
        })?;
        settings.save(SettingKind::Config, SETTINGS_APP.to_string(), Some(raw))
    }
}
