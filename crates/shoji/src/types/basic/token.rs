use std::fmt;

use serde::{Deserialize, Serialize};

#[bindings::export(Alias)]
pub type TokenId = u32;

#[bindings::export(Alias)]
pub type TokenValue = String;

#[bindings::export(ClassCloneable)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
