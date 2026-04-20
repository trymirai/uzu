use serde_json::Value;
use shoji::types::ContentBlockType;

pub struct ContentBlock {
    pub r#type: ContentBlockType,
    pub value: Value,
}
