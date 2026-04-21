use serde::{Deserialize, Serialize};

use crate::types::session::classification::Role;

#[bindings::export(Struct, name = "ClassificationMessage")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Message {
    pub role: Role,
    pub content: String,
}
