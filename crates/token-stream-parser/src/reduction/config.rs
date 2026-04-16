use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{reduction::ReductionParserError, types::TokenValue};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReductionParserGroup {
    Bounded {
        name: String,
        open_token: TokenValue,
        close_tokens: Vec<TokenValue>,
        #[serde(default)]
        groups: Vec<ReductionParserGroup>,
    },
    Open {
        name: String,
        open_token: TokenValue,
        #[serde(default)]
        groups: Vec<ReductionParserGroup>,
    },
    Greedy {
        name: String,
        #[serde(default)]
        capturing_limit: Option<usize>,
        #[serde(default)]
        groups: Vec<ReductionParserGroup>,
    },
}

impl ReductionParserGroup {
    pub fn name(&self) -> &str {
        match self {
            Self::Bounded {
                name,
                ..
            }
            | Self::Open {
                name,
                ..
            }
            | Self::Greedy {
                name,
                ..
            } => name,
        }
    }

    pub fn groups(&self) -> &[ReductionParserGroup] {
        match self {
            Self::Bounded {
                groups,
                ..
            }
            | Self::Open {
                groups,
                ..
            }
            | Self::Greedy {
                groups,
                ..
            } => groups,
        }
    }

    pub fn open_token(&self) -> Option<&TokenValue> {
        match self {
            Self::Bounded {
                open_token,
                ..
            }
            | Self::Open {
                open_token,
                ..
            } => Some(open_token),
            Self::Greedy {
                ..
            } => None,
        }
    }

    pub fn capturing_limit(&self) -> Option<usize> {
        match self {
            Self::Bounded {
                ..
            }
            | Self::Open {
                ..
            } => None,
            Self::Greedy {
                capturing_limit,
                ..
            } => *capturing_limit,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReductionParserConfig {
    pub groups: Vec<ReductionParserGroup>,
}

impl ReductionParserConfig {
    pub fn validate(&self) -> Result<(), ReductionParserError> {
        validate_sibling_groups(&self.groups)
    }

    pub fn collect_sections_compose_groups(&self) -> std::collections::HashSet<String> {
        let mut result = std::collections::HashSet::new();
        Self::collect_sections_compose_groups_recursive(&self.groups, &mut result);
        result
    }

    fn collect_sections_compose_groups_recursive(
        groups: &[ReductionParserGroup],
        result: &mut std::collections::HashSet<String>,
    ) {
        for group in groups {
            let children = group.groups();
            let has_non_greedy_children = !children.is_empty()
                && children.iter().all(|child| !matches!(child, ReductionParserGroup::Greedy { .. }));
            if has_non_greedy_children {
                result.insert(group.name().to_string());
            }
            Self::collect_sections_compose_groups_recursive(group.groups(), result);
        }
    }

    pub fn collect_framing_tokens(&self) -> Vec<TokenValue> {
        let mut tokens = Vec::new();
        Self::collect_framing_tokens_recursive(&self.groups, &mut tokens);
        tokens
    }

    fn collect_framing_tokens_recursive(
        groups: &[ReductionParserGroup],
        tokens: &mut Vec<TokenValue>,
    ) {
        for group in groups {
            if let Some(open) = group.open_token() {
                if !tokens.contains(open) {
                    tokens.push(open.clone());
                }
            }
            if let ReductionParserGroup::Bounded {
                close_tokens,
                ..
            } = group
            {
                for close_token in close_tokens {
                    if !tokens.contains(close_token) {
                        tokens.push(close_token.clone());
                    }
                }
            }
            Self::collect_framing_tokens_recursive(group.groups(), tokens);
        }
    }
}

fn validate_sibling_groups(groups: &[ReductionParserGroup]) -> Result<(), ReductionParserError> {
    let mut seen_names: HashSet<&str> = HashSet::new();
    let mut seen_open_tokens: HashSet<&TokenValue> = HashSet::new();

    for group in groups {
        if !seen_names.insert(group.name()) {
            return Err(ReductionParserError::DuplicateGroupName {
                name: group.name().to_string(),
            });
        }

        if let Some(open_token) = group.open_token() {
            if !seen_open_tokens.insert(open_token) {
                return Err(ReductionParserError::DuplicateOpenToken {
                    token: open_token.clone(),
                });
            }
        }

        validate_sibling_groups(group.groups())?;
    }

    Ok(())
}
