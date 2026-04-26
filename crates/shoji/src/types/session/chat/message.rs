use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{
    basic::{ReasoningEffort, ToolCall, ToolNamespace, Value},
    session::chat::{ChatContentBlock, ChatRole},
};

#[bindings::export(Structure)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ChatMessageMetadata {
    pub values: HashMap<String, Value>,
}

impl From<HashMap<String, Value>> for ChatMessageMetadata {
    fn from(values: HashMap<String, Value>) -> Self {
        Self {
            values,
        }
    }
}

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: Vec<ChatContentBlock>,
    pub metadata: ChatMessageMetadata,
}

#[bindings::export(Implementation)]
impl ChatMessage {
    #[bindings::export(Method(Factory))]
    pub fn for_role(role: ChatRole) -> Self {
        Self {
            role,
            content: vec![],
            metadata: ChatMessageMetadata {
                values: HashMap::new(),
            },
        }
    }

    #[bindings::export(Method(Factory))]
    pub fn system() -> Self {
        Self::for_role(ChatRole::System {})
    }

    #[bindings::export(Method(Factory))]
    pub fn developer() -> Self {
        Self::for_role(ChatRole::Developer {})
    }

    #[bindings::export(Method(Factory))]
    pub fn user() -> Self {
        Self::for_role(ChatRole::User {})
    }

    #[bindings::export(Method(Factory))]
    pub fn assistant() -> Self {
        Self::for_role(ChatRole::Assistant {})
    }

    #[bindings::export(Method(Factory))]
    pub fn tool() -> Self {
        Self::for_role(ChatRole::Tool {})
    }
}

#[bindings::export(Implementation)]
impl ChatMessage {
    #[bindings::export(Method)]
    pub fn with_block(
        &self,
        block: ChatContentBlock,
    ) -> Self {
        let mut content = self.content.clone();
        content.push(block);
        Self {
            content,
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_reasoning_effort(
        &self,
        reasoning_effort: ReasoningEffort,
    ) -> Self {
        self.with_block(ChatContentBlock::ReasoningEffort {
            value: reasoning_effort,
        })
    }

    #[bindings::export(Method)]
    pub fn with_tool_namespaces(
        &self,
        tool_namespaces: Vec<ToolNamespace>,
    ) -> Self {
        self.with_block(ChatContentBlock::Tools {
            namespaces: tool_namespaces,
        })
    }

    #[bindings::export(Method)]
    pub fn with_text(
        &self,
        text: String,
    ) -> Self {
        self.with_block(ChatContentBlock::Text {
            value: text,
        })
    }

    #[bindings::export(Method)]
    pub fn with_reasoning(
        &self,
        reasoning: String,
    ) -> Self {
        self.with_block(ChatContentBlock::Reasoning {
            value: reasoning,
        })
    }

    #[bindings::export(Method)]
    pub fn with_tool_call(
        &self,
        tool_call: ToolCall,
    ) -> Self {
        self.with_block(ChatContentBlock::ToolCall {
            value: tool_call,
        })
    }

    #[bindings::export(Method)]
    pub fn with_tool_call_candidate(
        &self,
        value: Value,
    ) -> Self {
        self.with_block(ChatContentBlock::ToolCallCandidate {
            value,
        })
    }
}

macro_rules! blocks_by_type {
    ($self:expr, $variant:ident, $field:ident) => {
        $self.content.iter().filter_map(|block| match block {
            ChatContentBlock::$variant {
                $field,
            } => Some($field.clone()),
            _ => None,
        })
    };
}

#[bindings::export(Implementation)]
impl ChatMessage {
    #[bindings::export(Method(Getter))]
    pub fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        blocks_by_type!(self, ReasoningEffort, value).next()
    }

    #[bindings::export(Method(Getter))]
    pub fn text(&self) -> Option<String> {
        blocks_by_type!(self, Text, value).reduce(|mut text, value| {
            text.push_str(&value);
            text
        })
    }

    #[bindings::export(Method(Getter))]
    pub fn reasoning(&self) -> Option<String> {
        blocks_by_type!(self, Reasoning, value).reduce(|mut reasoning, value| {
            reasoning.push_str(&value);
            reasoning
        })
    }

    #[bindings::export(Method(Getter))]
    pub fn tool_namespaces(&self) -> Vec<ToolNamespace> {
        blocks_by_type!(self, Tools, namespaces).flatten().collect()
    }

    #[bindings::export(Method(Getter))]
    pub fn tool_calls(&self) -> Vec<ToolCall> {
        blocks_by_type!(self, ToolCall, value).collect()
    }
}

impl ChatMessage {
    pub fn tool_call_results(&self) -> Vec<(Option<String>, Option<String>, Value)> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ChatContentBlock::ToolCallResult {
                    identifier,
                    name,
                    value,
                } => Some((identifier.clone(), name.clone(), value.clone())),
                _ => None,
            })
            .collect()
    }
}

pub trait ChatMessageList {
    fn reasoning_effort(&self) -> Option<ReasoningEffort>;
    fn tool_namespaces(&self) -> Vec<ToolNamespace>;
}

impl ChatMessageList for Vec<ChatMessage> {
    fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.iter().find_map(|message| message.reasoning_effort())
    }

    fn tool_namespaces(&self) -> Vec<ToolNamespace> {
        self.iter().flat_map(|message| message.tool_namespaces()).collect()
    }
}
