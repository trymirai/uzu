use serde_json::Value;
use shoji::types::session::chat::ChatContentBlockType;

pub struct ContentBlock {
    pub r#type: ChatContentBlockType,
    pub value: Value,
}
