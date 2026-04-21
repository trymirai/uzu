use serde::{Deserialize, Serialize};

#[bindings::export(Struct, name = "Repository")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Repository {
    pub identifier: String,
    pub commit_hash: String,
    pub paths: Option<Vec<String>>,
}
