use shoji::types::basic::ReasoningEffort;

use crate::{
    common::model_capabilities::{ThinkingPreference, ThinkingSupport},
    interactive::util::cycle,
};

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

/// Settings-UI behavior for [ThinkingSupport]: cycling through the adjustable
/// values, writing the choice back to preferences, and display labels.
pub trait ThinkingSupportExt {
    fn cycled(
        self,
        delta: i64,
    ) -> ThinkingSupport;
    fn write_back(
        self,
        preference: &mut ThinkingPreference,
    );
    fn value_label(self) -> &'static str;
}

impl ThinkingSupportExt for ThinkingSupport {
    fn cycled(
        self,
        delta: i64,
    ) -> ThinkingSupport {
        match self {
            ThinkingSupport::Levels(effort) => ThinkingSupport::Levels(cycle(&LEVELS, effort, delta as isize)),
            ThinkingSupport::Toggle(value) => ThinkingSupport::Toggle(!value),
            other => other,
        }
    }

    fn write_back(
        self,
        preference: &mut ThinkingPreference,
    ) {
        match self {
            ThinkingSupport::Levels(effort) => preference.level = effort,
            ThinkingSupport::Toggle(value) => preference.enabled = value,
            ThinkingSupport::AlwaysOn | ThinkingSupport::Unsupported => {},
        }
    }

    fn value_label(self) -> &'static str {
        match self {
            ThinkingSupport::Levels(effort) => level_label(effort),
            ThinkingSupport::Toggle(true) => "on",
            ThinkingSupport::Toggle(false) => "off",
            ThinkingSupport::AlwaysOn => "always on",
            ThinkingSupport::Unsupported => "not supported",
        }
    }
}
