mod jinja;
mod jinja_function;

use indexmap::IndexMap;
pub use jinja::JinjaConfig;
pub use jinja_function::JinjaFunction;
use serde::{Deserialize, Serialize};
use shoji::types::session::chat::{ChatContentBlockType, ChatRole};

use crate::chat::hanashi::messages::{
    canonical::Config as CanonicalConfig,
    rendered::{Config as RenderedConfig, Field, FieldConfig},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RendererConfig {
    pub jinja: JinjaConfig,
    pub canonization: CanonicalConfig,
    pub rendering: IndexMap<ChatRole, RenderedConfig>,
}

impl RendererConfig {
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
