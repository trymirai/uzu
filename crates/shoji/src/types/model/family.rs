use serde::{Deserialize, Serialize};

use crate::types::{basic::Metadata, model::ModelVendor};

#[bindings::export(ClassCloneable)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelFamily {
    #[serde(rename = "id")]
    pub identifier: String,
    pub vendor: ModelVendor,
    pub metadata: Metadata,
}

#[bindings::export(Implementation)]
impl ModelFamily {
    #[bindings::export(Getter)]
    pub fn name(&self) -> String {
        self.metadata.name.clone()
    }
}
