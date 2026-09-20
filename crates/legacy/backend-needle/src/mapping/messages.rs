use shoji::types::session::chat::{ChatContentBlock, ChatMessage, ChatRole};

use crate::error::Error;

#[derive(Debug, Clone, PartialEq)]
pub enum CompleteInput {
    User(String),
    ToolResults(Vec<serde_json::Value>),
}

pub fn system_text(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .filter(|message| message.role == ChatRole::System {})
        .filter_map(ChatMessage::text)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("; ")
}

pub fn check_unsupported(messages: &[ChatMessage]) -> Result<(), Error> {
    for message in messages {
        for block in &message.content {
            match block {
                ChatContentBlock::Image {
                    ..
                }
                | ChatContentBlock::Video {
                    ..
                }
                | ChatContentBlock::Audio {
                    ..
                }
                | ChatContentBlock::File {
                    ..
                } => return Err(Error::UnsupportedContent),
                _ => {},
            }
        }
    }
    Ok(())
}

pub fn complete_inputs(messages: &[ChatMessage]) -> Result<Vec<CompleteInput>, Error> {
    check_unsupported(messages)?;
    let mut inputs = Vec::new();
    let mut pending_tools: Vec<serde_json::Value> = Vec::new();

    let flush_tools = |inputs: &mut Vec<CompleteInput>, pending: &mut Vec<serde_json::Value>| {
        if !pending.is_empty() {
            inputs.push(CompleteInput::ToolResults(std::mem::take(pending)));
        }
    };

    for message in messages {
        match message.role {
            ChatRole::User {} => {
                flush_tools(&mut inputs, &mut pending_tools);
                if let Some(text) = message.text()
                    && !text.is_empty()
                {
                    inputs.push(CompleteInput::User(text));
                }
            },
            ChatRole::Tool {} => {
                for (_id, _name, value) in message.tool_call_results() {
                    let parsed = serde_json::from_str(&value.json)?;
                    pending_tools.push(parsed);
                }
            },
            ChatRole::Assistant {} => {
                flush_tools(&mut inputs, &mut pending_tools);
            },
            ChatRole::System {} | ChatRole::Developer {} => {},
            ChatRole::Custom {
                ref name,
            } => {
                tracing::warn!(name, "skipping custom chat role for Needle");
            },
        }
    }
    flush_tools(&mut inputs, &mut pending_tools);
    Ok(inputs)
}

impl CompleteInput {
    pub fn as_complete_text(&self) -> Result<String, Error> {
        match self {
            Self::User(text) => Ok(text.clone()),
            Self::ToolResults(values) => Ok(serde_json::to_string(values)?),
        }
    }
}
