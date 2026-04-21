use serde::{Deserialize, Serialize};

#[bindings::export(Enum, name = "ChatGrammar")]
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
