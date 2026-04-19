use std::collections::HashMap;

use openai_harmony::chat::{
    Author as ExternalAuthor, Content as ExternalContent, DeveloperContent as ExternalDeveloperContent,
    Message as ExternalMessage, Role as ExternalRole, SystemContent as ExternalSystemContent, TextContent,
};
use shoji::types::{ContentBlock, Message, ReasoningEffort, Role, ToolCall, ToolNamespace, Value};

use crate::chat::harmony::bridging::{Error, FromHarmony, ToHarmony};

const FUNCTIONS_NAMESPACE: &str = "functions";
const CHANNEL_ANALYSIS: &str = "analysis";
const CHANNEL_COMMENTARY: &str = "commentary";
const CHANNEL_FINAL: &str = "final";
const CONTENT_TYPE_JSON: &str = "json";
const RECIPIENT_ASSISTANT: &str = "assistant";
const BUILTIN_BROWSER: &str = "browser";
const BUILTIN_PYTHON: &str = "python";

pub fn bridge_messages_to_harmony(messages: &[Message]) -> Result<Vec<ExternalMessage>, Error> {
    let mut result = Vec::new();

    for message in messages {
        match &message.role {
            Role::System {} => {
                let mut system_filled = false;
                let mut system_content = ExternalSystemContent::default();
                let mut text_parts: Vec<String> = Vec::new();

                for block in &message.content {
                    match block {
                        ContentBlock::Identity {
                            value,
                        } => {
                            system_filled = true;
                            system_content.model_identity = Some(value.clone());
                        },
                        ContentBlock::ReasoningEffort {
                            value,
                        } => {
                            system_filled = true;
                            system_content.reasoning_effort = Some(value.clone().to_harmony()?);
                        },
                        ContentBlock::ConversationStartDate {
                            value,
                        } => {
                            system_filled = true;
                            system_content.conversation_start_date = Some(value.clone());
                        },
                        ContentBlock::KnowledgeCutoff {
                            value,
                        } => {
                            system_filled = true;
                            system_content.knowledge_cutoff = Some(value.clone());
                        },
                        ContentBlock::BuiltinTools {
                            names,
                        } => {
                            for name in names {
                                match name.as_str() {
                                    BUILTIN_BROWSER => {
                                        system_filled = true;
                                        system_content = system_content.with_browser_tool();
                                    },
                                    BUILTIN_PYTHON => {
                                        system_filled = true;
                                        system_content = system_content.with_python_tool();
                                    },
                                    _ => {
                                        return Err(Error::UnsupportedBuiltinTool {
                                            name: name.clone(),
                                        });
                                    },
                                }
                            }
                        },
                        ContentBlock::Text {
                            value,
                        } => {
                            text_parts.push(value.clone());
                        },
                        other => {
                            return Err(Error::UnsupportedContentBlock {
                                block_type: other.get_type(),
                                role: message.role.clone(),
                            });
                        },
                    }
                }

                let mut content: Vec<ExternalContent> = vec![];
                if system_filled {
                    content.push(ExternalContent::SystemContent(system_content));
                }
                if !text_parts.is_empty() {
                    content.push(ExternalContent::Text(TextContent {
                        text: text_parts.join(""),
                    }));
                }

                result.push(ExternalMessage {
                    author: ExternalAuthor::from(ExternalRole::System),
                    recipient: None,
                    content,
                    channel: None,
                    content_type: None,
                });
            },
            Role::Developer {} => {
                let mut developer_content = ExternalDeveloperContent::default();
                let mut text_parts: Vec<String> = Vec::new();

                for block in &message.content {
                    match block {
                        ContentBlock::Text {
                            value,
                        } => {
                            text_parts.push(value.clone());
                        },
                        ContentBlock::Tools {
                            namespaces,
                        } => {
                            for namespace in namespaces {
                                developer_content = developer_content.with_tools(namespace.clone().to_harmony()?);
                            }
                        },
                        other => {
                            return Err(Error::UnsupportedContentBlock {
                                block_type: other.get_type(),
                                role: message.role.clone(),
                            });
                        },
                    }
                }

                if !text_parts.is_empty() {
                    developer_content.instructions = Some(text_parts.join(""));
                }

                result.push(ExternalMessage {
                    author: ExternalAuthor::from(ExternalRole::Developer),
                    recipient: None,
                    content: vec![ExternalContent::DeveloperContent(developer_content)],
                    channel: None,
                    content_type: None,
                });
            },
            Role::User {} => {
                let mut text_parts: Vec<String> = Vec::new();

                for block in &message.content {
                    match block {
                        ContentBlock::Text {
                            value,
                        } => {
                            text_parts.push(value.clone());
                        },
                        other => {
                            return Err(Error::UnsupportedContentBlock {
                                block_type: other.get_type(),
                                role: message.role.clone(),
                            });
                        },
                    }
                }

                if text_parts.is_empty() {
                    return Err(Error::ContentRequired {
                        role: message.role.clone(),
                    });
                }

                result.push(ExternalMessage::from_role_and_content(ExternalRole::User, text_parts.join("")));
            },
            Role::Assistant {} => {
                let mut reasoning_parts: Vec<String> = Vec::new();
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_calls: Vec<ToolCall> = Vec::new();

                for block in &message.content {
                    match block {
                        ContentBlock::Reasoning {
                            value,
                        } => {
                            reasoning_parts.push(value.clone());
                        },
                        ContentBlock::Text {
                            value,
                        } => {
                            text_parts.push(value.clone());
                        },
                        ContentBlock::ToolCall {
                            value,
                        } => {
                            tool_calls.push(value.clone());
                        },
                        other => {
                            return Err(Error::UnsupportedContentBlock {
                                block_type: other.get_type(),
                                role: message.role.clone(),
                            });
                        },
                    }
                }

                if tool_calls.len() > 1 {
                    return Err(Error::MultipleToolCalls);
                }

                if !reasoning_parts.is_empty() {
                    result.push(
                        ExternalMessage::from_role_and_content(ExternalRole::Assistant, reasoning_parts.join(""))
                            .with_channel(CHANNEL_ANALYSIS),
                    );
                }

                if !text_parts.is_empty() {
                    result.push(
                        ExternalMessage::from_role_and_content(ExternalRole::Assistant, text_parts.join(""))
                            .with_channel(CHANNEL_FINAL),
                    );
                }

                if let Some(tool_call) = tool_calls.first() {
                    let arguments = serde_json::Value::try_from(tool_call.arguments.clone()).map_err(|error| {
                        Error::SerializationFailed {
                            message: error.to_string(),
                        }
                    })?;

                    result.push(
                        ExternalMessage::from_role_and_content(
                            ExternalRole::Assistant,
                            serde_json::to_string(&arguments).map_err(|error| Error::SerializationFailed {
                                message: error.to_string(),
                            })?,
                        )
                        .with_channel(CHANNEL_COMMENTARY)
                        .with_recipient(format!("{FUNCTIONS_NAMESPACE}.{}", tool_call.name))
                        .with_content_type(CONTENT_TYPE_JSON),
                    );
                }
            },
            Role::Tool {} => {
                if message.content.len() != 1 {
                    return Err(Error::MultipleContentBlocks);
                }

                let block = &message.content[0];
                let ContentBlock::ToolCallResult {
                    name,
                    value,
                } = block
                else {
                    return Err(Error::UnsupportedContentBlock {
                        block_type: block.get_type(),
                        role: message.role.clone(),
                    });
                };

                let name = name.as_ref().ok_or(Error::MissingToolCallResultName)?;

                let result_value =
                    serde_json::Value::try_from(value.clone()).map_err(|error| Error::SerializationFailed {
                        message: error.to_string(),
                    })?;

                result.push(
                    ExternalMessage::from_author_and_content(
                        ExternalAuthor::new(ExternalRole::Tool, format!("{FUNCTIONS_NAMESPACE}.{name}")),
                        serde_json::to_string(&result_value).map_err(|error| Error::SerializationFailed {
                            message: error.to_string(),
                        })?,
                    )
                    .with_channel(CHANNEL_COMMENTARY)
                    .with_recipient(RECIPIENT_ASSISTANT),
                );
            },
            Role::Custom {
                ..
            } => {
                return Err(Error::UnsupportedRole {
                    role: message.role.clone(),
                });
            },
        }
    }

    Ok(result)
}

pub fn bridge_messages_from_harmony(messages: &[ExternalMessage]) -> Result<Vec<Message>, Error> {
    let mut result = Vec::new();
    let mut index = 0;

    while index < messages.len() {
        let message = &messages[index];

        match &message.author.role {
            ExternalRole::System => {
                let mut content = Vec::new();

                for external_content in &message.content {
                    match external_content {
                        ExternalContent::SystemContent(system_content) => {
                            if let Some(identity) = &system_content.model_identity {
                                content.push(ContentBlock::Identity {
                                    value: identity.clone(),
                                });
                            }
                            if let Some(effort) = &system_content.reasoning_effort {
                                content.push(ContentBlock::ReasoningEffort {
                                    value: ReasoningEffort::from_harmony(*effort),
                                });
                            }
                            if let Some(date) = &system_content.conversation_start_date {
                                content.push(ContentBlock::ConversationStartDate {
                                    value: date.clone(),
                                });
                            }
                            if let Some(cutoff) = &system_content.knowledge_cutoff {
                                content.push(ContentBlock::KnowledgeCutoff {
                                    value: cutoff.clone(),
                                });
                            }
                            if let Some(tools) = &system_content.tools {
                                let names: Vec<String> = tools.keys().cloned().collect();
                                if !names.is_empty() {
                                    content.push(ContentBlock::BuiltinTools {
                                        names,
                                    });
                                }
                            }
                        },
                        ExternalContent::Text(text_content) => {
                            content.push(ContentBlock::Text {
                                value: text_content.text.clone(),
                            });
                        },
                        _ => {
                            return Err(Error::SerializationFailed {
                                message: "Unexpected content type in system message".to_string(),
                            });
                        },
                    }
                }

                result.push(Message {
                    role: Role::System {},
                    content,
                    metadata: HashMap::new(),
                });
                index += 1;
            },
            ExternalRole::Developer => {
                let mut content = Vec::new();

                for external_content in &message.content {
                    match external_content {
                        ExternalContent::DeveloperContent(developer_content) => {
                            if let Some(instructions) = &developer_content.instructions {
                                content.push(ContentBlock::Text {
                                    value: instructions.clone(),
                                });
                            }
                            if let Some(tools) = &developer_content.tools {
                                let namespaces: Vec<ToolNamespace> =
                                    tools.values().cloned().map(ToolNamespace::from_harmony).collect();
                                if !namespaces.is_empty() {
                                    content.push(ContentBlock::Tools {
                                        namespaces,
                                    });
                                }
                            }
                        },
                        ExternalContent::Text(text_content) => {
                            content.push(ContentBlock::Text {
                                value: text_content.text.clone(),
                            });
                        },
                        _ => {
                            return Err(Error::SerializationFailed {
                                message: "Unexpected content type in developer message".to_string(),
                            });
                        },
                    }
                }

                result.push(Message {
                    role: Role::Developer {},
                    content,
                    metadata: HashMap::new(),
                });
                index += 1;
            },
            ExternalRole::User => {
                let text = extract_text_content(&message.content);

                result.push(Message {
                    role: Role::User {},
                    content: vec![ContentBlock::Text {
                        value: text,
                    }],
                    metadata: HashMap::new(),
                });
                index += 1;
            },
            ExternalRole::Assistant => {
                let mut content = Vec::new();

                while index < messages.len() && messages[index].author.role == ExternalRole::Assistant {
                    let assistant_message = &messages[index];
                    let text = extract_text_content(&assistant_message.content);

                    match assistant_message.channel.as_deref() {
                        Some(CHANNEL_ANALYSIS) => {
                            content.push(ContentBlock::Reasoning {
                                value: text,
                            });
                        },
                        Some(CHANNEL_COMMENTARY) => {
                            let recipient =
                                assistant_message.recipient.as_deref().ok_or_else(|| Error::SerializationFailed {
                                    message: "Tool call missing recipient".to_string(),
                                })?;
                            let name = strip_namespace_prefix(recipient);

                            match serde_json::from_str::<serde_json::Value>(&text) {
                                Ok(json_value) => {
                                    content.push(ContentBlock::ToolCall {
                                        value: ToolCall {
                                            name: name.to_string(),
                                            arguments: Value::from(json_value),
                                        },
                                    });
                                },
                                Err(_) => {
                                    content.push(ContentBlock::ToolCallCandidate {
                                        value: Value::from(serde_json::Value::String(text)),
                                    });
                                },
                            }
                        },
                        _ => {
                            content.push(ContentBlock::Text {
                                value: text,
                            });
                        },
                    }

                    index += 1;
                }

                result.push(Message {
                    role: Role::Assistant {},
                    content,
                    metadata: HashMap::new(),
                });
            },
            ExternalRole::Tool => {
                let text = extract_text_content(&message.content);
                let qualified_name = message.author.name.as_deref().ok_or(Error::MissingToolCallResultName)?;
                let name = strip_namespace_prefix(qualified_name);

                let value = Value::from(serde_json::from_str::<serde_json::Value>(&text).map_err(|error| {
                    Error::SerializationFailed {
                        message: error.to_string(),
                    }
                })?);

                result.push(Message {
                    role: Role::Tool {},
                    content: vec![ContentBlock::ToolCallResult {
                        name: Some(name.to_string()),
                        value,
                    }],
                    metadata: HashMap::new(),
                });
                index += 1;
            },
        }
    }

    Ok(result)
}

fn extract_text_content(content: &[ExternalContent]) -> String {
    content
        .iter()
        .filter_map(|external_content| match external_content {
            ExternalContent::Text(text_content) => Some(text_content.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn strip_namespace_prefix(qualified_name: &str) -> &str {
    qualified_name.strip_prefix(FUNCTIONS_NAMESPACE).and_then(|rest| rest.strip_prefix('.')).unwrap_or(qualified_name)
}
