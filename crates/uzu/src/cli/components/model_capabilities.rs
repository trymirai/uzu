use std::path::Path;

use serde_json::Value;
use shoji::types::{basic::ReasoningEffort, model::Model};

use super::preferences::{ThinkingPreference, cycle};
use crate::engine::Engine;

const LEVELS: [ReasoningEffort; 5] = [
    ReasoningEffort::Default,
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
    ReasoningEffort::Disabled,
];

fn level_label(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Default => "model default",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Disabled => "off",
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

    pub fn cycled(
        self,
        delta: i64,
    ) -> Self {
        match self {
            Self::Levels(effort) => Self::Levels(cycle(&LEVELS, effort, delta as isize)),
            Self::Toggle(value) => Self::Toggle(!value),
            other => other,
        }
    }

    pub fn write_back(
        self,
        preference: &mut ThinkingPreference,
    ) {
        match self {
            Self::Levels(effort) => preference.level = effort,
            Self::Toggle(value) => preference.enabled = value,
            Self::AlwaysOn | Self::Unsupported => {},
        }
    }

    pub fn value_label(self) -> &'static str {
        match self {
            Self::Levels(effort) => level_label(effort),
            Self::Toggle(true) => "on",
            Self::Toggle(false) => "off",
            Self::AlwaysOn => "always on",
            Self::Unsupported => "not supported",
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
    use crate::cli::components::preferences::ThinkingPreference;

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
}
