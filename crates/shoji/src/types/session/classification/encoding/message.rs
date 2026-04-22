use serde::{Deserialize, Serialize};

use crate::types::session::classification::Role;

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(
        role: Role,
        content: String,
    ) -> Self {
        Self {
            role,
            content,
        }
    }

    pub fn user(content: String) -> Self {
        Self::new(Role::User, content)
    }

    pub fn assistant(content: String) -> Self {
        Self::new(Role::Assistant, content)
    }
}
