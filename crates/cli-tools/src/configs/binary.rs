use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct BinaryConfig {
    #[serde(rename = "crate")]
    pub crate_name: String,
    pub targets: Vec<String>,
}
