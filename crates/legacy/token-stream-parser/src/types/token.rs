use std::fmt;

use serde::{Deserialize, Serialize};

pub type TokenId = u32;
pub type TokenValue = String;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Token {
    pub id: TokenId,
    pub value: TokenValue,
    pub is_special: bool,
}

impl fmt::Display for Token {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let escaped: String = self.value.chars().flat_map(|character| character.escape_debug()).collect();
        write!(formatter, "{escaped}")
    }
}
