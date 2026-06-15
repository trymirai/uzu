use serde::{Deserialize, Serialize};
use shoji::types::basic::{ReasoningEffort, SamplingMethod, SamplingPolicy};

use crate::settings::{SettingKind, Settings, SettingsError};

const SETTINGS_PREFERENCES: &str = "cli_preferences";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ThinkingPreference {
    pub level: ReasoningEffort,
    pub enabled: bool,
}

impl Default for ThinkingPreference {
    fn default() -> Self {
        Self {
            level: ReasoningEffort::Default,
            enabled: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SamplingMode {
    #[default]
    ModelDefault,
    Greedy,
    Stochastic,
}

impl SamplingMode {
    pub const ALL: [SamplingMode; 3] = [Self::ModelDefault, Self::Greedy, Self::Stochastic];

    pub fn label(&self) -> &'static str {
        match self {
            Self::ModelDefault => "model default",
            Self::Greedy => "greedy",
            Self::Stochastic => "stochastic",
        }
    }

    pub fn next(self) -> Self {
        cycle(&Self::ALL, self, 1)
    }

    pub fn previous(self) -> Self {
        cycle(&Self::ALL, self, -1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SamplingPreferences {
    pub mode: SamplingMode,
    pub temperature_enabled: bool,
    pub temperature: f64,
    pub top_k_enabled: bool,
    pub top_k: i64,
    pub top_p_enabled: bool,
    pub top_p: f64,
    pub min_p_enabled: bool,
    pub min_p: f64,
    pub repetition_penalty_enabled: bool,
    pub repetition_penalty: f64,
    pub suffix_repetition_length: i64,
}

impl SamplingPreferences {
    pub fn suffix_repetition_length_enabled(&self) -> bool {
        self.repetition_penalty_enabled
    }

    pub fn suffix_repetition_length_disable(&mut self) {
        self.repetition_penalty_enabled = false;
    }

    pub fn policy(&self) -> SamplingPolicy {
        match self.mode {
            SamplingMode::ModelDefault => SamplingPolicy::Default {},
            SamplingMode::Greedy => SamplingPolicy::Custom {
                method: SamplingMethod::Greedy {},
            },
            SamplingMode::Stochastic => SamplingPolicy::Custom {
                method: SamplingMethod::Stochastic {
                    temperature: self.temperature_enabled.then_some(self.temperature),
                    top_k: self.top_k_enabled.then_some(self.top_k),
                    top_p: self.top_p_enabled.then_some(self.top_p),
                    min_p: self.min_p_enabled.then_some(self.min_p),
                    repetition_penalty: self.repetition_penalty_enabled.then_some(self.repetition_penalty),
                    suffix_repetition_length: self
                        .suffix_repetition_length_enabled()
                        .then_some(self.suffix_repetition_length),
                },
            },
        }
    }

    pub fn summary(&self) -> String {
        match self.mode {
            SamplingMode::ModelDefault => "default".to_string(),
            SamplingMode::Greedy => "greedy".to_string(),
            SamplingMode::Stochastic => {
                let mut parts = Vec::new();
                if self.temperature_enabled {
                    parts.push(format!("temp {:.2}", self.temperature));
                }
                if self.top_k_enabled {
                    parts.push(format!("top-k {}", self.top_k));
                }
                if self.top_p_enabled {
                    parts.push(format!("top-p {:.2}", self.top_p));
                }
                if self.min_p_enabled {
                    parts.push(format!("min-p {:.2}", self.min_p));
                }
                if self.repetition_penalty_enabled {
                    parts.push(format!("repetition penalty {:.2}", self.repetition_penalty));
                }
                if self.suffix_repetition_length_enabled() {
                    parts.push(format!("suffix repetition length {:.2}", self.suffix_repetition_length));
                }
                if parts.is_empty() {
                    "stochastic".to_string()
                } else {
                    parts.join(", ")
                }
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Preferences {
    pub thinking: ThinkingPreference,
    pub sampling: SamplingPreferences,
}

impl Preferences {
    pub fn load(settings: &Settings) -> Result<Self, SettingsError> {
        let Some(raw) = settings.load(SettingKind::Config, SETTINGS_PREFERENCES.to_string())? else {
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
        settings.save(SettingKind::Config, SETTINGS_PREFERENCES.to_string(), Some(raw))
    }

    pub fn sampling_policy(&self) -> SamplingPolicy {
        self.sampling.policy()
    }
}

pub(super) fn cycle<T: Copy + PartialEq>(
    values: &[T],
    current: T,
    delta: isize,
) -> T {
    let length = values.len() as isize;
    let index = values.iter().position(|value| *value == current).map(|index| index as isize).unwrap_or(0);
    let next = (index + delta).rem_euclid(length) as usize;
    values[next]
}
