use serde::{Deserialize, Serialize};
use shoji::types::basic::{ReasoningEffort, SamplingMethod, SamplingPolicy};

use crate::settings::{SettingKind, Settings, SettingsError};

const SETTINGS_PREFERENCES: &str = "cli_preferences";

/// How much reasoning ("thinking") the model is asked to perform before answering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingMode {
    /// Leave the decision to the model's chat template.
    #[default]
    ModelDefault,
    /// Explicitly disable reasoning.
    Off,
    Low,
    Medium,
    High,
}

impl ThinkingMode {
    pub const ALL: [ThinkingMode; 5] = [Self::ModelDefault, Self::Off, Self::Low, Self::Medium, Self::High];

    pub fn label(&self) -> &'static str {
        match self {
            Self::ModelDefault => "model default",
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    pub fn next(self) -> Self {
        cycle(&Self::ALL, self, 1)
    }

    pub fn previous(self) -> Self {
        cycle(&Self::ALL, self, -1)
    }

    /// Reasoning effort to attach to a chat message, or `None` to leave the
    /// template default untouched.
    pub fn reasoning_effort(self) -> Option<ReasoningEffort> {
        match self {
            Self::ModelDefault => None,
            Self::Off => Some(ReasoningEffort::Disabled),
            Self::Low => Some(ReasoningEffort::Low),
            Self::Medium => Some(ReasoningEffort::Medium),
            Self::High => Some(ReasoningEffort::High),
        }
    }
}

/// Which sampling strategy is used when generating replies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SamplingMode {
    /// Use whatever the backend considers default.
    #[default]
    ModelDefault,
    /// Always pick the most likely token.
    Greedy,
    /// Sample from the distribution shaped by the parameters below.
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
}

impl Default for SamplingPreferences {
    fn default() -> Self {
        Self {
            mode: SamplingMode::default(),
            temperature_enabled: false,
            temperature: 0.8,
            top_k_enabled: false,
            top_k: 64,
            top_p_enabled: false,
            top_p: 0.95,
            min_p_enabled: false,
            min_p: 0.05,
        }
    }
}

impl SamplingPreferences {
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
                },
            },
        }
    }

    /// Short, human-readable summary of the active sampling configuration.
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
pub struct Preferences {
    pub thinking: ThinkingMode,
    pub sampling: SamplingPreferences,
}

impl Preferences {
    pub fn load(settings: &Settings) -> Result<Self, SettingsError> {
        let Some(raw) = settings.load(SettingKind::Config, SETTINGS_PREFERENCES.to_string())? else {
            return Ok(Self::default());
        };
        // Tolerate older / partially-written payloads by falling back to defaults.
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

    pub fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.thinking.reasoning_effort()
    }

    pub fn sampling_policy(&self) -> SamplingPolicy {
        self.sampling.policy()
    }
}

fn cycle<T: Copy + PartialEq>(
    values: &[T],
    current: T,
    delta: isize,
) -> T {
    let length = values.len() as isize;
    let index = values.iter().position(|value| *value == current).map(|index| index as isize).unwrap_or(0);
    let next = (index + delta).rem_euclid(length) as usize;
    values[next]
}
