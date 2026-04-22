use serde::{Deserialize, Serialize};

#[bindings::export(Enum)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChatContextLength {
    Default {},
    Maximal {},
    Custom {
        length: i64,
    },
}

impl Default for ChatContextLength {
    fn default() -> Self {
        ChatContextLength::Default {}
    }
}
