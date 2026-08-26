use shoji::types::basic::ReasoningEffort;

use crate::common::thinking::{ThinkingPreference, ThinkingSupport};

fn level_label(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Default => "model default",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::XHigh => "xhigh",
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
            ThinkingSupport::Levels {
                effort,
                supported,
            } => {
                let levels = supported.iter().collect::<Vec<_>>();
                let effort = crate::interactive::util::cycle(&levels, effort, delta as isize);
                ThinkingSupport::Levels {
                    effort,
                    supported,
                }
            },
            ThinkingSupport::Toggle(value) => ThinkingSupport::Toggle(!value),
            other => other,
        }
    }

    fn write_back(
        self,
        preference: &mut ThinkingPreference,
    ) {
        match self {
            ThinkingSupport::Levels {
                effort,
                ..
            } => preference.level = effort,
            ThinkingSupport::Toggle(value) => preference.enabled = value,
            ThinkingSupport::AlwaysOn | ThinkingSupport::Unsupported => {},
        }
    }

    fn value_label(self) -> &'static str {
        match self {
            ThinkingSupport::Levels {
                effort,
                ..
            } => level_label(effort),
            ThinkingSupport::Toggle(true) => "on",
            ThinkingSupport::Toggle(false) => "off",
            ThinkingSupport::AlwaysOn => "always on",
            ThinkingSupport::Unsupported => "not supported",
        }
    }
}
