use serde::{Deserialize, Serialize};
use serde_json::Value;
use shoji::types::session::chat::{ChatContentBlock, ChatRole, ToolCall};

use crate::chat::hanashi::messages::streamed::Section;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Sections(Vec<Section>),
    Value(Value),
}

impl Content {
    pub fn blocks(
        self,
        role: &ChatRole,
    ) -> Vec<ChatContentBlock> {
        match self {
            Content::Text(text) => vec![ChatContentBlock::Text {
                value: text,
            }],
            Content::Value(value) => match role {
                ChatRole::Tool {} => vec![ChatContentBlock::ToolCallResult {
                    identifier: None,
                    name: None,
                    value: value.into(),
                }],
                _ => vec![ChatContentBlock::Custom {
                    value: value.into(),
                }],
            },
            Content::Sections(sections) => {
                let mut reasoning_parts: Vec<String> = Vec::new();
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_calls: Vec<ChatContentBlock> = Vec::new();

                for section in sections {
                    match section {
                        Section::Text {
                            value: Some(text),
                        } => text_parts.push(text),
                        Section::Reasoning {
                            value: Some(text),
                        } => reasoning_parts.push(text),
                        Section::ToolCall {
                            value: Some(value),
                        } => {
                            let block = match serde_json::from_value::<ToolCall>(value.clone()) {
                                Ok(tool_call) => ChatContentBlock::ToolCall {
                                    value: tool_call,
                                },
                                Err(_) => ChatContentBlock::ToolCallCandidate {
                                    value: value.into(),
                                },
                            };
                            tool_calls.push(block);
                        },
                        Section::ToolCall {
                            value: None,
                        } => {
                            tool_calls.push(ChatContentBlock::ToolCallCandidate {
                                value: Value::Null.into(),
                            });
                        },
                        _ => {},
                    }
                }

                let reasoning = reasoning_parts.concat().trim().to_string();
                let text = text_parts.concat().trim().to_string();
                let mut blocks = Vec::new();
                if !reasoning.is_empty() {
                    blocks.push(ChatContentBlock::Reasoning {
                        value: reasoning,
                    });
                }
                if !text.is_empty() {
                    blocks.push(ChatContentBlock::Text {
                        value: text,
                    });
                }
                blocks.extend(tool_calls);
                blocks
            },
        }
    }
}
