use serde::{Deserialize, Serialize};

use crate::types::basic::Metadata;

#[bindings::export(ClassCloneable)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelBackend {
    #[serde(rename = "id")]
    pub identifier: String,
    pub version: String,
    pub metadata: Metadata,
}
