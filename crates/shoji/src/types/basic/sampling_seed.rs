use serde::{Deserialize, Serialize};

#[bindings::export(Enum)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SamplingSeed {
    Default {},
    Custom {
        seed: i64,
    },
}

impl Default for SamplingSeed {
    fn default() -> Self {
        SamplingSeed::Default {}
    }
}
