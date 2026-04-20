use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{
    basic::Value,
    encoding::{ContentBlock, ReasoningEffort, Role, ToolCall, ToolNamespace},
};

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    pub metadata: HashMap<String, Value>,
}

impl Message {
    pub fn for_role(role: Role) -> Self {
        Self {
            role,
            content: vec![],
            metadata: HashMap::new(),
        }
    }

    pub fn system() -> Self {
        Self::for_role(Role::System {})
    }

    pub fn developer() -> Self {
        Self::for_role(Role::Developer {})
    }

    pub fn user() -> Self {
        Self::for_role(Role::User {})
    }

    pub fn assistant() -> Self {
        Self::for_role(Role::Assistant {})
    }

    pub fn tool() -> Self {
        Self::for_role(Role::Tool {})
    }
}

impl Message {
    pub fn with_reasoning_effort(
        self,
        reasoning_effort: ReasoningEffort,
    ) -> Self {
        Self {
            content: vec![ContentBlock::ReasoningEffort {
                value: reasoning_effort,
            }],
            ..self
        }
    }

    pub fn with_tool_namespaces(
        self,
        tool_namespaces: Vec<ToolNamespace>,
    ) -> Self {
        Self {
            content: vec![ContentBlock::Tools {
                namespaces: tool_namespaces,
            }],
            ..self
        }
    }

    pub fn with_text(
        self,
        text: String,
    ) -> Self {
        Self {
            content: vec![ContentBlock::Text {
                value: text,
            }],
            ..self
        }
    }

    pub fn with_reasoning(
        self,
        reasoning: String,
    ) -> Self {
        Self {
            content: vec![ContentBlock::Reasoning {
                value: reasoning,
            }],
            ..self
        }
    }
}

macro_rules! blocks_by_type {
    ($self:expr, $variant:ident, $field:ident) => {
        $self.content.iter().filter_map(|block| match block {
            ContentBlock::$variant {
                $field,
            } => Some($field.clone()),
            _ => None,
        })
    };
}

impl Message {
    pub fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        blocks_by_type!(self, ReasoningEffort, value).next()
    }

    pub fn text(&self) -> String {
        blocks_by_type!(self, Text, value).collect::<String>()
    }

    pub fn reasoning(&self) -> String {
        blocks_by_type!(self, Reasoning, value).collect::<String>()
    }

    pub fn tool_namespaces(&self) -> Vec<ToolNamespace> {
        blocks_by_type!(self, Tools, namespaces).flatten().collect()
    }

    pub fn tool_calls(&self) -> Vec<ToolCall> {
        blocks_by_type!(self, ToolCall, value).collect()
    }

    pub fn tool_call_results(&self) -> Vec<(Option<String>, Value)> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolCallResult {
                    name,
                    value,
                } => Some((name.clone(), value.clone())),
                _ => None,
            })
            .collect()
    }
}

pub trait MessageList {
    fn reasoning_effort(&self) -> Option<ReasoningEffort>;
    fn tool_namespaces(&self) -> Vec<ToolNamespace>;
}

impl MessageList for Vec<Message> {
    fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.iter().find_map(|message| message.reasoning_effort())
    }

    fn tool_namespaces(&self) -> Vec<ToolNamespace> {
        self.iter().flat_map(|message| message.tool_namespaces()).collect()
    }
}
