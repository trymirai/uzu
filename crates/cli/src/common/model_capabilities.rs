use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use shoji::types::{basic::ReasoningEffort, model::Model};
use uzu::engine::Engine;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingSupport {
    Levels(ReasoningEffort),
    Toggle(bool),
    AlwaysOn,
    #[default]
    Unsupported,
}

impl ThinkingSupport {
    pub fn is_adjustable(self) -> bool {
        matches!(self, Self::Levels(_) | Self::Toggle(_))
    }

    pub fn with_preference(
        self,
        preference: &ThinkingPreference,
    ) -> Self {
        match self {
            Self::Levels(_) => Self::Levels(preference.level),
            Self::Toggle(_) => Self::Toggle(preference.enabled),
            other => other,
        }
    }

    /// Resolve a requested effort against what this model can actually do.
    /// `Ok(Some(effort))` — emit this effort; `Ok(None)` — the request is already
    /// satisfied by the model's fixed behavior; `Err` — the requested thinking
    /// state cannot be produced, so the caller must reject the request rather
    /// than silently ignore it.
    pub fn fulfill_requested_effort(
        self,
        effort: ReasoningEffort,
    ) -> Result<Option<ReasoningEffort>, String> {
        match self {
            // A levels model applies the exact effort; default needs no explicit message.
            Self::Levels(_) => Ok(if effort == ReasoningEffort::Default {
                None
            } else {
                Some(effort)
            }),
            // A toggle model honors the on/off state; effort levels collapse to on.
            Self::Toggle(_) => Ok(Some(if effort == ReasoningEffort::Disabled {
                ReasoningEffort::Disabled
            } else {
                ReasoningEffort::Default
            })),
            Self::AlwaysOn => {
                if effort == ReasoningEffort::Disabled {
                    Err("model always reasons; reasoning cannot be disabled".to_string())
                } else {
                    Ok(None)
                }
            },
            Self::Unsupported => {
                if effort == ReasoningEffort::Disabled {
                    Ok(None)
                } else {
                    Err("model does not support reasoning".to_string())
                }
            },
        }
    }

    pub fn reasoning_effort(self) -> Option<ReasoningEffort> {
        match self {
            Self::Levels(ReasoningEffort::Default) => None,
            Self::Levels(effort) => Some(effort),
            Self::Toggle(true) => Some(ReasoningEffort::Default),
            Self::Toggle(false) => Some(ReasoningEffort::Disabled),
            Self::AlwaysOn | Self::Unsupported => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ModelSamplingDefaults {
    pub temperature: Option<f64>,
    pub top_k: Option<i64>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub suffix_repetition_length: Option<i64>,
}

impl ModelSamplingDefaults {
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(value) = self.temperature {
            parts.push(format!("temp {value:.2}"));
        }
        if let Some(value) = self.top_k {
            parts.push(format!("top-k {value}"));
        }
        if let Some(value) = self.top_p {
            parts.push(format!("top-p {value:.2}"));
        }
        if let Some(value) = self.min_p {
            parts.push(format!("min-p {value:.2}"));
        }
        if let Some(value) = self.repetition_penalty {
            parts.push(format!("repetition penalty {:.2}", value));
        }
        if let Some(value) = self.suffix_repetition_length {
            parts.push(format!("suffix repetition length {:.2}", value));
        }
        if parts.is_empty() {
            "model defaults".to_string()
        } else {
            parts.join(", ")
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ModelCapabilities {
    pub thinking: ThinkingSupport,
    pub sampling_defaults: ModelSamplingDefaults,
}

impl ModelCapabilities {
    pub async fn load(
        engine: &Engine,
        model: &Model,
    ) -> Self {
        if model.is_remote() {
            return Self {
                thinking: ThinkingSupport::Levels(ReasoningEffort::Default),
                sampling_defaults: ModelSamplingDefaults::default(),
            };
        }
        let Some(path) = engine.model_path(model).await else {
            return Self::default();
        };
        let raw = tokio::fs::read_to_string(Path::new(&path).join("config.json")).await.ok();
        raw.and_then(|raw| Self::from_config(&raw)).unwrap_or_default()
    }

    fn from_config(raw: &str) -> Option<Self> {
        let json: Value = serde_json::from_str(raw).ok()?;

        let codec = json.get("token_codec_config");
        let template = codec.and_then(|codec| codec.get("prompt_template")).and_then(Value::as_str).unwrap_or_default();
        let emits_reasoning =
            codec.and_then(|codec| codec.get("output_parser_regex")).map(|value| !value.is_null()).unwrap_or(false);
        let thinking = if template.contains("enable_thinking") {
            ThinkingSupport::Toggle(true)
        } else if emits_reasoning {
            ThinkingSupport::AlwaysOn
        } else {
            ThinkingSupport::Unsupported
        };

        let generation = json.get("generation_config");
        let field = |key: &str| generation.and_then(|generation| generation.get(key));
        let sampling_defaults = ModelSamplingDefaults {
            temperature: field("temperature").and_then(Value::as_f64),
            top_k: field("top_k").and_then(Value::as_i64),
            top_p: field("top_p").and_then(Value::as_f64),
            min_p: field("min_p").and_then(Value::as_f64),
            repetition_penalty: field("repetition_penalty").and_then(Value::as_f64),
            suffix_repetition_length: field("suffix_repetition_length").and_then(Value::as_i64),
        };

        Some(Self {
            thinking,
            sampling_defaults,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_emits_explicit_effort_each_state() {
        assert_eq!(ThinkingSupport::Toggle(true).reasoning_effort(), Some(ReasoningEffort::Default));
        assert_eq!(ThinkingSupport::Toggle(false).reasoning_effort(), Some(ReasoningEffort::Disabled));
    }

    #[test]
    fn toggle_round_trips_through_preference() {
        let support = ThinkingSupport::Toggle(true);
        let enabled = ThinkingPreference {
            enabled: true,
            ..ThinkingPreference::default()
        };
        let disabled = ThinkingPreference {
            enabled: false,
            ..ThinkingPreference::default()
        };
        assert_eq!(support.with_preference(&enabled).reasoning_effort(), Some(ReasoningEffort::Default));
        assert_eq!(support.with_preference(&disabled).reasoning_effort(), Some(ReasoningEffort::Disabled));
    }

    #[test]
    fn non_adjustable_support_emits_no_effort() {
        assert_eq!(ThinkingSupport::AlwaysOn.reasoning_effort(), None);
        assert_eq!(ThinkingSupport::Unsupported.reasoning_effort(), None);
        assert_eq!(ThinkingSupport::Levels(ReasoningEffort::Default).reasoning_effort(), None);
        assert_eq!(ThinkingSupport::Levels(ReasoningEffort::High).reasoning_effort(), Some(ReasoningEffort::High));
    }

    #[test]
    fn fulfill_requested_effort_honors_adjustable_models() {
        assert_eq!(
            ThinkingSupport::Levels(ReasoningEffort::Default).fulfill_requested_effort(ReasoningEffort::High),
            Ok(Some(ReasoningEffort::High))
        );
        assert_eq!(
            ThinkingSupport::Levels(ReasoningEffort::Default).fulfill_requested_effort(ReasoningEffort::Default),
            Ok(None)
        );
        assert_eq!(
            ThinkingSupport::Toggle(true).fulfill_requested_effort(ReasoningEffort::Low),
            Ok(Some(ReasoningEffort::Default))
        );
        assert_eq!(
            ThinkingSupport::Toggle(true).fulfill_requested_effort(ReasoningEffort::Disabled),
            Ok(Some(ReasoningEffort::Disabled))
        );
    }

    #[test]
    fn fulfill_requested_effort_rejects_what_fixed_models_cannot_do() {
        assert_eq!(ThinkingSupport::AlwaysOn.fulfill_requested_effort(ReasoningEffort::High), Ok(None));
        assert!(ThinkingSupport::AlwaysOn.fulfill_requested_effort(ReasoningEffort::Disabled).is_err());
        assert_eq!(ThinkingSupport::Unsupported.fulfill_requested_effort(ReasoningEffort::Disabled), Ok(None));
        assert!(ThinkingSupport::Unsupported.fulfill_requested_effort(ReasoningEffort::Low).is_err());
    }
}
