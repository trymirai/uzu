use serde::{Deserialize, Serialize};

use crate::registry::types::Image;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EntityType {
    Vendor,
    Family,
    Model,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Entity {
    pub r#type: EntityType,
    pub identifier: String,
    pub name: String,
    pub description: Option<String>,
    pub icon: Vec<Image>,
}
