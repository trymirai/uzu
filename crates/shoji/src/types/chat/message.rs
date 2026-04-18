use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{ContentBlock, Role, Value};

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    pub metadata: HashMap<String, Value>,
}
