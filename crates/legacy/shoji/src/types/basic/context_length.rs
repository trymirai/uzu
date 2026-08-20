use serde::{Deserialize, Serialize};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContextLength {
    Default {},
    Maximal {},
    Custom {
        length: i64,
    },
}

impl Default for ContextLength {
    fn default() -> Self {
        ContextLength::Default {}
    }
}
