use serde::{Deserialize, Serialize};

use crate::types::basic::Metadata;

#[bindings::export(ClassCloneable)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelProperties {
    #[serde(rename = "id")]
    pub identifier: String,
    pub size: i64,
    pub version: Option<String>,
    pub metadata: Metadata,
}

#[bindings::export(Implementation)]
impl ModelProperties {
    #[bindings::export(Getter)]
    pub fn name(&self) -> String {
        self.metadata.name.clone()
    }
}
