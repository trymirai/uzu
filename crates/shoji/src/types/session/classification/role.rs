use serde::{Deserialize, Serialize};

#[bindings::export(Enum)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClassificationRole {
    User,
    Assistant,
}
