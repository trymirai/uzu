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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Context {
    pub tokenizer_location: TokenizerLocation,
}
