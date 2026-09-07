use std::{fs, io, path::Path};

use serde::{Deserialize, Serialize};

use crate::{
    common::thinking::ThinkingPreference,
    interactive::{components::Theme, sampling::SamplingPreferences},
};

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Preferences {
    pub theme: Theme,
    pub thinking: ThinkingPreference,
    pub sampling: SamplingPreferences,
    pub selected_model_id: Option<String>,
    #[cfg(all(target_os = "macos", feature = "hardware-control"))]
    #[serde(skip)]
    pub performance_mode: crate::interactive::hardware::PerformanceMode,
}

impl Preferences {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        Ok(toml::from_str(&contents)?)
    }

    pub fn store(
        &self,
        path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let parent = path
            .parent()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "settings file has no parent directory"))?;
        fs::create_dir_all(parent)?;
        fs::write(path, toml::to_string(self)?)?;
        Ok(())
    }
}
