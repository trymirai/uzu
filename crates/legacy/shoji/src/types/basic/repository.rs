use serde::{Deserialize, Serialize};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Repository {
    pub identifier: String,
    pub commit_hash: Option<String>,
    pub paths: Option<Vec<String>>,
}
