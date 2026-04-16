use std::fmt;

use serde::{Deserialize, Serialize};
use token_stream_parser::types::Token as ParserToken;

pub type TokenId = u32;
pub type TokenValue = String;

#[bindings::export(Struct)]
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

impl From<ParserToken> for Token {
    fn from(token: ParserToken) -> Self {
        Self {
            id: token.id,
            value: token.value,
            is_special: token.is_special,
        }
    }
}

impl From<Token> for ParserToken {
    fn from(token: Token) -> Self {
        Self {
            id: token.id,
            value: token.value,
            is_special: token.is_special,
        }
    }
}
