use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CheckedFileState {
    Valid,
    Invalid,
    Missing,
}
