use serde::{Deserialize, Serialize};

use crate::types::basic::Image;

#[bindings::export(Enum)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelEntityType {
    Registry,
    Backend,
    Vendor,
    Family,
    Variant,
}

#[bindings::export(ClassCloneable)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelEntity {
    pub r#type: ModelEntityType,
    pub identifier: String,
    pub parent_identifier: Option<String>,
    pub name: String,
    pub description: Option<String>,
    pub version: Option<String>,
    pub icons: Vec<Image>,
}
