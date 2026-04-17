use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub type_names: IndexMap<String, String>,
    pub role_key: String,
    pub type_key: String,
    pub text_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url_key: Option<String>,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub custom_keys: IndexMap<String, String>,
}
