use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{
    basic::Value,
    session::chat::{ContentBlock, ReasoningEffort, Role, ToolCall, ToolNamespace},
};

#[bindings::export(Struct, name = "Message")]
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
    fn with_block(
        self,
        block: ContentBlock,
    ) -> Self {
        let mut content = self.content;
        content.push(block);
        Self {
            content,
            ..self
        }
    }

    pub fn with_reasoning_effort(
        self,
        reasoning_effort: ReasoningEffort,
    ) -> Self {
        self.with_block(ContentBlock::ReasoningEffort {
            value: reasoning_effort,
        })
    }

    pub fn with_tool_namespaces(
        self,
        tool_namespaces: Vec<ToolNamespace>,
    ) -> Self {
        self.with_block(ContentBlock::Tools {
            namespaces: tool_namespaces,
        })
    }

    pub fn with_text(
        self,
        text: String,
    ) -> Self {
        self.with_block(ContentBlock::Text {
            value: text,
        })
    }

    pub fn with_reasoning(
        self,
        reasoning: String,
    ) -> Self {
        self.with_block(ContentBlock::Reasoning {
            value: reasoning,
        })
    }

    pub fn with_tool_call(
        self,
        tool_call: ToolCall,
    ) -> Self {
        self.with_block(ContentBlock::ToolCall {
            value: tool_call,
        })
    }

    pub fn with_tool_call_candidate(
        self,
        value: Value,
    ) -> Self {
        self.with_block(ContentBlock::ToolCallCandidate {
            value,
        })
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

    pub fn text(&self) -> Option<String> {
        blocks_by_type!(self, Text, value).reduce(|mut text, value| {
            text.push_str(&value);
            text
        })
    }

    pub fn reasoning(&self) -> Option<String> {
        blocks_by_type!(self, Reasoning, value).reduce(|mut reasoning, value| {
            reasoning.push_str(&value);
            reasoning
        })
    }

    pub fn tool_namespaces(&self) -> Vec<ToolNamespace> {
        blocks_by_type!(self, Tools, namespaces).flatten().collect()
    }

    pub fn tool_calls(&self) -> Vec<ToolCall> {
        blocks_by_type!(self, ToolCall, value).collect()
    }

    pub fn tool_call_results(&self) -> Vec<(Option<String>, Option<String>, Value)> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolCallResult {
                    identifier,
                    name,
                    value,
                } => Some((identifier.clone(), name.clone(), value.clone())),
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
