use serde::{Deserialize, Serialize};

#[bindings::export(Struct, name = "ChatFeature")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Feature {
    pub name: String,
    pub values: Vec<String>,
}
