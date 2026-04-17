use serde_json::Value;

use crate::chat::types::ContentBlockType;

pub struct ContentBlock {
    pub r#type: ContentBlockType,
    pub value: Value,
}
