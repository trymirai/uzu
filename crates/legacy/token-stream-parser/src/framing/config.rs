use serde::{Deserialize, Serialize};

use crate::types::TokenValue;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FramingParserConfig {
    pub tokens: Vec<TokenValue>,
}
