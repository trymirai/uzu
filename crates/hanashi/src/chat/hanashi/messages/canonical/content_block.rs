use serde_json::Value;
use shoji::types::encoding::ContentBlockType;

pub struct ContentBlock {
    pub r#type: ContentBlockType,
    pub value: Value,
}
