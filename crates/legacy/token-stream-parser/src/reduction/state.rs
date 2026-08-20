use serde::{Deserialize, Serialize};

use crate::{ParserState, framing::FramingParserSection, types::Token};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReductionParserSection {
    Frame(FramingParserSection),
    Group {
        name: String,
        #[serde(default)]
        open: Option<Token>,
        #[serde(default)]
        close: Option<Token>,
        #[serde(default)]
        finished: bool,
        #[serde(default)]
        sections: Vec<ReductionParserSection>,
    },
}

impl ParserState for ReductionParserSection {
    fn is_substate_of(
        &self,
        other_state: &Self,
    ) -> bool {
        match (self, other_state) {
            (Self::Frame(current), Self::Frame(other)) => current.is_substate_of(other),
            (
                Self::Group {
                    name: current_name,
                    open: current_open,
                    sections: current_sections,
                    ..
                },
                Self::Group {
                    name: other_name,
                    open: other_open,
                    sections: other_sections,
                    ..
                },
            ) => {
                current_name == other_name
                    && current_open == other_open
                    && sections_is_substate(current_sections, other_sections)
            },
            _ => false,
        }
    }

    fn tokens(&self) -> Vec<&Token> {
        match self {
            Self::Frame(section) => section.tokens(),
            Self::Group {
                open,
                close,
                sections,
                ..
            } => {
                let mut result = Vec::new();
                if let Some(token) = open {
                    result.push(token);
                }
                for section in sections {
                    result.extend(section.tokens());
                }
                if let Some(token) = close {
                    result.push(token);
                }
                result
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReductionParserState {
    pub sections: Vec<ReductionParserSection>,
}

impl ReductionParserState {
    pub fn new() -> Self {
        Self {
            sections: Vec::new(),
        }
    }
}

impl ParserState for ReductionParserState {
    fn is_substate_of(
        &self,
        other_state: &Self,
    ) -> bool {
        sections_is_substate(&self.sections, &other_state.sections)
    }

    fn tokens(&self) -> Vec<&Token> {
        self.sections.iter().flat_map(|section| section.tokens()).collect()
    }
}

fn sections_is_substate(
    current: &[ReductionParserSection],
    other: &[ReductionParserSection],
) -> bool {
    if current.is_empty() {
        return true;
    }
    if current.len() > other.len() {
        return false;
    }
    for (index, section) in current.iter().enumerate() {
        if index < current.len() - 1 {
            if section != &other[index] {
                return false;
            }
        } else if !section.is_substate_of(&other[index]) {
            return false;
        }
    }
    true
}
