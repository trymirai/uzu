use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use shoji::types::encoding::ContentBlockType;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Field {
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub disable_raw: bool,
    #[serde(flatten)]
    pub config: FieldConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FieldConfig {
    Unique {
        block: ContentBlockType,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mapping: Option<IndexMap<String, Option<Value>>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        allowed_values: Option<Vec<Value>>,
    },
    Collected {
        blocks: Vec<ContentBlockType>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        limit: Option<usize>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub role: String,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub message: IndexMap<String, Field>,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub context: IndexMap<String, Field>,
}
