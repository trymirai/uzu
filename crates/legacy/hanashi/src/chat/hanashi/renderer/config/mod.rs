mod jinja;
mod jinja_function;

use indexmap::IndexMap;
pub use jinja::JinjaConfig;
pub use jinja_function::JinjaFunction;
use serde::{Deserialize, Serialize};
use shoji::types::{
    basic::ReasoningEffort,
    session::chat::{ChatContentBlockType, ChatRole},
};

use crate::chat::hanashi::messages::{
    canonical::Config as CanonicalConfig,
    rendered::{Config as RenderedConfig, Field, FieldConfig},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RendererConfig {
    pub jinja: JinjaConfig,
    pub canonization: CanonicalConfig,
    pub rendering: IndexMap<ChatRole, RenderedConfig>,
    // Needed when the default lives inside the template rather than its field mapping.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_reasoning_effort: Option<ReasoningEffort>,
}

impl RendererConfig {
    pub fn default_reasoning_effort(&self) -> Option<ReasoningEffort> {
        if let Some(effort) = self.default_reasoning_effort {
            return Some(effort);
        }

        let mappings: Option<Vec<_>> = self
            .rendering
            .values()
            .flat_map(|role| role.message.values().chain(role.context.values()))
            .filter_map(|field| match &field.config {
                FieldConfig::Unique {
                    block: ChatContentBlockType::ReasoningEffort,
                    mapping,
                    ..
                } => Some(mapping.as_ref()),
                _ => None,
            })
            .collect();
        let mappings = mappings.filter(|mappings| !mappings.is_empty())?;

        // Null omits a template variable; matching two null mappings does not
        // identify the template's fallback as either of those named modes.
        if !mappings.iter().any(|mapping| matches!(mapping.get("default"), Some(Some(_)))) {
            return None;
        }

        // A control may span several fields (Qwen3.8 has both enable_thinking
        // and reasoning_effort). Only an exact match across all of them proves
        // that an explicit effort has the same meaning as the default.
        let mut matches = [
            ReasoningEffort::Disabled,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::XHigh,
        ]
        .into_iter()
        .filter(|effort| {
            mappings.iter().all(|mapping| {
                mapping.get(&effort.to_string()).is_some_and(|value| Some(value) == mapping.get("default"))
            })
        });
        if let Some(effort) = matches.next() {
            return matches.next().is_none().then_some(effort);
        }

        // Toggle-only templates name their enabled mode "default". An omitted
        // or null default still depends on the template and cannot prove this.
        (mappings
            .iter()
            .all(|mapping| matches!(mapping.get("default"), Some(Some(_))) && mapping.contains_key("disabled"))
            && mappings.iter().any(|mapping| mapping.get("default") != mapping.get("disabled")))
        .then_some(ReasoningEffort::Default)
    }

    pub fn get_role_by_name(
        &self,
        name: &str,
    ) -> ChatRole {
        self.rendering
            .iter()
            .find(|(_, rendered_config)| rendered_config.role == name)
            .map(|(role, _)| role.clone())
            .unwrap_or_else(|| ChatRole::Custom {
                name: name.to_string(),
            })
    }

    pub fn get_rendering_role_and_field_for_block_type(
        &self,
        block_type: &ChatContentBlockType,
    ) -> Option<(&ChatRole, &Field)> {
        self.rendering.iter().find_map(|(role, role_config)| {
            role_config
                .message
                .values()
                .chain(role_config.context.values())
                .find(|field| match &field.config {
                    FieldConfig::Unique {
                        block,
                        ..
                    } => block == block_type,
                    FieldConfig::Collected {
                        blocks,
                        ..
                    } => blocks.contains(block_type),
                })
                .map(|field| (role, field))
        })
    }

    pub fn get_rendering_field_for_block_type(
        &self,
        block_type: &ChatContentBlockType,
    ) -> Option<&Field> {
        self.get_rendering_role_and_field_for_block_type(block_type).map(|(_, field)| field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::hanashi::config::HanashiConfig;

    #[test]
    fn omitted_default_and_disabled_do_not_prove_the_template_default_is_off() {
        let mut config = HanashiConfig::Qwen35.resolve().unwrap().rendering;
        let field = config.rendering.get_mut(&ChatRole::System {}).unwrap().context.get_mut("enable_thinking").unwrap();
        let FieldConfig::Unique {
            mapping: Some(mapping),
            ..
        } = &mut field.config
        else {
            panic!("expected a reasoning mapping");
        };
        mapping.insert("default".to_string(), None);
        mapping.insert("disabled".to_string(), None);
        assert_eq!(config.default_reasoning_effort(), None);
    }

    #[test]
    fn default_effort_compares_every_control_field() {
        let config = HanashiConfig::Qwen38.resolve().unwrap().rendering;
        assert_eq!(config.default_reasoning_effort(), Some(ReasoningEffort::XHigh));

        let mut ambiguous = config;
        ambiguous.rendering.get_mut(&ChatRole::System {}).unwrap().context.shift_remove("reasoning_effort");
        // The remaining boolean field matches low, medium, and xhigh equally.
        assert_eq!(ambiguous.default_reasoning_effort(), None);
    }

    #[test]
    fn muse_glimmer_default_is_high() {
        let config = HanashiConfig::MuseGlimmer.resolve().unwrap().rendering;
        assert_eq!(config.default_reasoning_effort(), Some(ReasoningEffort::High));
    }

    #[test]
    fn toggle_default_requires_a_distinct_declared_value() {
        let mut config = HanashiConfig::Qwen35.resolve().unwrap().rendering;
        assert_eq!(config.default_reasoning_effort(), Some(ReasoningEffort::Default));
        let set_default = |config: &mut RendererConfig, value| {
            let field =
                config.rendering.get_mut(&ChatRole::System {}).unwrap().context.get_mut("enable_thinking").unwrap();
            let FieldConfig::Unique {
                mapping: Some(mapping),
                ..
            } = &mut field.config
            else {
                panic!("expected mapped control")
            };
            mapping.insert("default".to_string(), value);
        };
        set_default(&mut config, None);
        assert_eq!(config.default_reasoning_effort(), None);
        set_default(&mut config, Some(false.into()));
        assert_eq!(config.default_reasoning_effort(), Some(ReasoningEffort::Disabled));
    }

    #[test]
    fn template_fallback_requires_declared_default_metadata() {
        let mut config = HanashiConfig::GptOss.resolve().unwrap().rendering;
        assert_eq!(config.default_reasoning_effort(), Some(ReasoningEffort::Medium));
        config.default_reasoning_effort = None;
        assert_eq!(config.default_reasoning_effort(), None);
    }
}
