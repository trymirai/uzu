use hanashi::chat::EncodingConfig;
use serde::{Deserialize, Serialize};
use shoji::types::{basic::ReasoningEffort, model::Model};

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

const EFFORT_ORDER: [ReasoningEffort; 6] = [
    ReasoningEffort::Default,
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
    ReasoningEffort::XHigh,
    ReasoningEffort::Disabled,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningEffortSet(u8);

impl ReasoningEffortSet {
    pub const ALL: Self = Self(0b111111);

    fn bit(effort: ReasoningEffort) -> u8 {
        match effort {
            ReasoningEffort::Default => 1 << 0,
            ReasoningEffort::Low => 1 << 1,
            ReasoningEffort::Medium => 1 << 2,
            ReasoningEffort::High => 1 << 3,
            ReasoningEffort::XHigh => 1 << 4,
            ReasoningEffort::Disabled => 1 << 5,
        }
    }

    pub fn contains(
        self,
        effort: ReasoningEffort,
    ) -> bool {
        self.0 & Self::bit(effort) != 0
    }

    fn from_efforts(efforts: &[ReasoningEffort]) -> Self {
        efforts.iter().fold(Self(0), |set, effort| Self(set.0 | Self::bit(*effort)))
    }

    pub fn iter(self) -> impl Iterator<Item = ReasoningEffort> {
        EFFORT_ORDER.into_iter().filter(move |effort| self.contains(*effort))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingSupport {
    Levels {
        effort: ReasoningEffort,
        supported: ReasoningEffortSet,
    },
    Toggle(bool),
    AlwaysOn,
    #[default]
    Unsupported,
}

impl ThinkingSupport {
    pub fn for_model(model: &Model) -> Self {
        if model.is_remote() {
            return Self::Levels {
                effort: ReasoningEffort::Default,
                supported: ReasoningEffortSet::ALL,
            };
        }
        let Some(value) = model.encoding.as_ref() else {
            return Self::Unsupported;
        };
        let config = match serde_json::from_str::<EncodingConfig>(&value.json) {
            Ok(config) => config,
            Err(error) => {
                tracing::warn!(?error, model = %model.identifier, "ignoring invalid encoding config");
                return Self::Unsupported;
            },
        };
        match config.capabilities() {
            Ok(capabilities) if capabilities.supports_reasoning => {
                let supported = ReasoningEffortSet::from_efforts(&capabilities.reasoning_efforts);
                let has_levels = capabilities.reasoning_efforts.iter().any(|effort| {
                    matches!(
                        effort,
                        ReasoningEffort::Low | ReasoningEffort::Medium | ReasoningEffort::High | ReasoningEffort::XHigh
                    )
                });
                if has_levels {
                    Self::Levels {
                        effort: ReasoningEffort::Default,
                        supported,
                    }
                } else if capabilities.supports_disable_reasoning {
                    Self::Toggle(true)
                } else {
                    Self::AlwaysOn
                }
            },
            _ => Self::Unsupported,
        }
    }

    pub fn is_adjustable(self) -> bool {
        matches!(self, Self::Levels { .. } | Self::Toggle(_))
    }

    pub fn with_preference(
        self,
        preference: &ThinkingPreference,
    ) -> Self {
        match self {
            Self::Levels {
                supported,
                ..
            } => Self::Levels {
                effort: if supported.contains(preference.level) {
                    preference.level
                } else {
                    ReasoningEffort::Default
                },
                supported,
            },
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
            Self::Levels {
                supported,
                ..
            } => {
                if effort == ReasoningEffort::Default {
                    Ok(None)
                } else if supported.contains(effort) {
                    Ok(Some(effort))
                } else {
                    Err(format!("model does not support reasoning effort {effort}"))
                }
            },
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
            Self::Levels {
                effort: ReasoningEffort::Default,
                ..
            } => None,
            Self::Levels {
                effort,
                ..
            } => Some(effort),
            Self::Toggle(true) => Some(ReasoningEffort::Default),
            Self::Toggle(false) => Some(ReasoningEffort::Disabled),
            Self::AlwaysOn | Self::Unsupported => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use shoji::types::{
        basic::Value,
        model::{ModelAccessibility, ModelReference, ModelSpecialization},
    };

    use super::*;

    fn levels(
        effort: ReasoningEffort,
        supported: &[ReasoningEffort],
    ) -> ThinkingSupport {
        ThinkingSupport::Levels {
            effort,
            supported: ReasoningEffortSet::from_efforts(supported),
        }
    }

    const QWEN38_EFFORTS: [ReasoningEffort; 5] = [
        ReasoningEffort::Default,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::XHigh,
        ReasoningEffort::Disabled,
    ];

    fn model_with_encoding(encoding: Option<Value>) -> Model {
        Model::external(
            "test".to_string(),
            "test".to_string(),
            "test".to_string(),
            "uzu".to_string(),
            "uzu".to_string(),
            "0.0".to_string(),
            vec![ModelSpecialization::Chat {}],
            ModelAccessibility::Local {
                reference: ModelReference::Local {
                    path: "/nonexistent".to_string(),
                },
            },
            encoding,
        )
    }

    fn encoding(name: &str) -> Value {
        Value::from(serde_json::json!({"name": name, "type": "hanashi"}))
    }

    #[test]
    fn for_model_maps_encoding_capabilities() {
        assert_eq!(
            ThinkingSupport::for_model(&model_with_encoding(Some(encoding("qwen3.8")))),
            levels(ReasoningEffort::Default, &QWEN38_EFFORTS)
        );
        assert_eq!(
            ThinkingSupport::for_model(&model_with_encoding(Some(encoding("qwen3.5")))),
            ThinkingSupport::Toggle(true)
        );
        assert_eq!(
            ThinkingSupport::for_model(&model_with_encoding(Some(encoding("qwen3-thinking")))),
            ThinkingSupport::AlwaysOn
        );
        assert_eq!(
            ThinkingSupport::for_model(&model_with_encoding(Some(encoding("gemma-3")))),
            ThinkingSupport::Unsupported
        );
        assert_eq!(ThinkingSupport::for_model(&model_with_encoding(None)), ThinkingSupport::Unsupported);
    }

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
    fn levels_fall_back_to_default_for_unsupported_preference() {
        let support = levels(ReasoningEffort::Default, &QWEN38_EFFORTS);
        let preference = ThinkingPreference {
            level: ReasoningEffort::High,
            ..ThinkingPreference::default()
        };
        assert_eq!(support.with_preference(&preference).reasoning_effort(), None);
    }

    #[test]
    fn non_adjustable_support_emits_no_effort() {
        assert_eq!(ThinkingSupport::AlwaysOn.reasoning_effort(), None);
        assert_eq!(ThinkingSupport::Unsupported.reasoning_effort(), None);
        assert_eq!(levels(ReasoningEffort::Default, &QWEN38_EFFORTS).reasoning_effort(), None);
        assert_eq!(levels(ReasoningEffort::Low, &QWEN38_EFFORTS).reasoning_effort(), Some(ReasoningEffort::Low));
    }

    #[test]
    fn fulfill_requested_effort_honors_adjustable_models() {
        let support = levels(ReasoningEffort::Default, &QWEN38_EFFORTS);
        assert_eq!(support.fulfill_requested_effort(ReasoningEffort::XHigh), Ok(Some(ReasoningEffort::XHigh)));
        assert_eq!(support.fulfill_requested_effort(ReasoningEffort::Default), Ok(None));
        assert!(support.fulfill_requested_effort(ReasoningEffort::High).is_err());
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
