use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Repository {
    pub identifier: String,
    pub commit_hash: String,
    pub paths: Option<Vec<String>>,
}
