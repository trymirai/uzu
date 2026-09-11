use serde::{Deserialize, Serialize};

use crate::types::basic::Metadata;

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelBackend {
    #[serde(rename = "id")]
    pub identifier: String,
    pub version: String,
    pub metadata: Metadata,
}

#[bindings::export(Implementation)]
impl ModelBackend {
    #[bindings::export(Method(Getter))]
    pub fn name(&self) -> String {
        self.metadata.name.clone()
    }
}
