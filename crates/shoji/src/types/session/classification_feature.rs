use serde::{Deserialize, Serialize};

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClassificationFeature {
    pub name: String,
    pub values: Vec<String>,
}
