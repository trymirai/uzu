use serde::{Deserialize, Serialize};

use crate::types::basic::Metadata;

#[bindings::export(ClassCloneable)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelRegistry {
    #[serde(rename = "id")]
    pub identifier: String,
    pub metadata: Metadata,
}

#[bindings::export(Implementation)]
impl ModelRegistry {
    #[bindings::export(Getter)]
    pub fn name(&self) -> String {
        self.metadata.name.clone()
    }
}
