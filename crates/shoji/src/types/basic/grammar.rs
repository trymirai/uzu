use serde::{Deserialize, Serialize};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Grammar {
    JsonAny {},
    JsonSchema {
        schema: String,
    },
    Regex {
        pattern: String,
    },
}
