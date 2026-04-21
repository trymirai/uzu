use serde::{Deserialize, Serialize};

#[bindings::export(Enum, name = "ClassificationRole")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    User,
    Assistant,
}
