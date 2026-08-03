use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TokenizerLocation {
    Directory {
        path: String,
        name: Option<String>,
    },
    File {
        path: String,
    },
}
