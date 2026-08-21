use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{framing::FramingParserSection, types::Token};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FramingParserOutput {
    Added(FramingParserSection),
    Extended(Token),
}

impl fmt::Display for FramingParserOutput {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Added(section) => write!(formatter, "{section}"),
            Self::Extended(token) => {
                write!(formatter, "extend({})", token.value)
            },
        }
    }
}
