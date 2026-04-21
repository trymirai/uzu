use serde::{Deserialize, Serialize};

use crate::types::model::Image;

#[bindings::export(Enum)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Registry,
    Backend,
    Vendor,
    Family,
    Variant,
}

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Entity {
    pub r#type: EntityType,
    pub identifier: String,
    pub parent_identifier: Option<String>,
    pub name: String,
    pub description: Option<String>,
    pub version: Option<String>,
    pub icons: Vec<Image>,
}
