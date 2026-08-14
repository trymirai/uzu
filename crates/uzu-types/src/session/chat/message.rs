use std::{collections::HashMap, fs::Metadata};

use serde::{Deserialize, Serialize};

use crate::{
    common::{Role, Value},
    core::ContentPart,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ChatMessage {
    pub role: Role,
    pub content: Vec<ContentPart>,
    pub metadata: HashMap<String, Value>,
}

impl ChatMessage {
    pub fn new(role: Role) -> Self {
        Self {
            role,
            content: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_content(
        &self,
        block: ContentPart,
    ) -> Self {
        let mut content = self.content.clone();
        content.push(block);
        Self {
            content,
            ..self.clone()
        }
    }
}
