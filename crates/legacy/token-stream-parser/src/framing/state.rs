use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{ParserState, types::Token};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FramingParserSection {
    Marker(Token),
    Text(Vec<Token>),
}

impl fmt::Display for FramingParserSection {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Marker(token) => {
                write!(formatter, "frame.marker({})", token.value)
            },
            Self::Text(tokens) => {
                let text: String = tokens.iter().map(|token| token.value.as_str()).collect();
                let escaped: String = text.chars().flat_map(|character| character.escape_debug()).collect();
                write!(formatter, "frame.text({escaped})")
            },
        }
    }
}

impl ParserState for FramingParserSection {
    fn is_substate_of(
        &self,
        other_state: &Self,
    ) -> bool {
        match (self, other_state) {
            (Self::Marker(current), Self::Marker(other)) => current == other,
            (Self::Text(current_tokens), Self::Text(other_tokens)) => {
                current_tokens.len() <= other_tokens.len()
                    && current_tokens.iter().zip(other_tokens.iter()).all(|(current, other)| current == other)
            },
            _ => false,
        }
    }

    fn tokens(&self) -> Vec<&Token> {
        match self {
            Self::Marker(token) => vec![token],
            Self::Text(tokens) => tokens.iter().collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FramingParserState {
    pub sections: Vec<FramingParserSection>,
}

impl FramingParserState {
    pub fn new() -> Self {
        Self {
            sections: Vec::new(),
        }
    }
}

impl ParserState for FramingParserState {
    fn is_substate_of(
        &self,
        other_state: &Self,
    ) -> bool {
        self.sections.len() <= other_state.sections.len()
            && self.sections.iter().zip(other_state.sections.iter()).enumerate().all(|(index, (current, other))| {
                if index == self.sections.len() - 1 {
                    current.is_substate_of(other)
                } else {
                    current == other
                }
            })
    }

    fn tokens(&self) -> Vec<&Token> {
        self.sections.iter().flat_map(|section| section.tokens()).collect()
    }
}
