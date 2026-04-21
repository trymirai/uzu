use std::collections::HashMap;

use indexmap::IndexMap;
use serde_json::Value;
use shoji::types::session::chat::{
    ContentBlock as OriginalContentBlock, ContentBlockType, Message as OriginalMessage, Role, TranslationInput,
};

use crate::chat::hanashi::messages::{
    Error,
    canonical::{Config, ContentBlock},
};

pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    pub metadata: HashMap<String, Value>,
}

impl Message {
    pub fn from_message(
        message: &OriginalMessage,
        config: &Config,
        raw_types: &[ContentBlockType],
    ) -> Result<Self, Error> {
        let mut content = Vec::new();
        for block in &message.content {
            let block_type = block.get_type();
            let raw = raw_types.contains(&block_type);
            let canonical_block = ContentBlock {
                r#type: block_type,
                value: Self::canonicalize_block(block, config, raw)?,
            };
            content.push(canonical_block);
        }

        Ok(Self {
            role: message.role.clone(),
            content,
            metadata: message
                .metadata
                .iter()
                .map(|(key, value)| {
                    let json_value: Value =
                        value.clone().try_into().map_err(|error: serde_json::Error| Error::SerializationFailed {
                            reason: error.to_string(),
                        })?;
                    Ok((key.clone(), json_value))
                })
                .collect::<Result<_, Error>>()?,
        })
    }

    fn canonicalize_block(
        block: &OriginalContentBlock,
        config: &Config,
        raw: bool,
    ) -> Result<Value, Error> {
        let type_name =
            config.type_names.get(&block.get_type().to_string()).cloned().unwrap_or(block.get_type().to_string());

        let value = match block {
            OriginalContentBlock::Identity {
                value,
            } => serde_json::to_value(value.clone()),
            OriginalContentBlock::ReasoningEffort {
                value,
            } => serde_json::to_value(value.to_string()),
            OriginalContentBlock::ConversationStartDate {
                value,
            } => serde_json::to_value(value.clone()),
            OriginalContentBlock::KnowledgeCutoff {
                value,
            } => serde_json::to_value(value.clone()),
            OriginalContentBlock::BuiltinTools {
                names,
            } => serde_json::to_value(names),
            OriginalContentBlock::Tools {
                namespaces,
            } => {
                let tools = namespaces.iter().flat_map(|namespace| namespace.tools.clone()).collect::<Vec<_>>();
                serde_json::to_value(tools)
            },
            OriginalContentBlock::Text {
                value,
            } => {
                if raw {
                    serde_json::to_value(value.clone())
                } else {
                    serde_json::to_value(IndexMap::from([
                        (config.type_key.clone(), type_name),
                        (config.text_key.clone(), value.clone()),
                    ]))
                }
            },
            OriginalContentBlock::Image {
                url,
            }
            | OriginalContentBlock::Video {
                url,
            }
            | OriginalContentBlock::Audio {
                url,
            }
            | OriginalContentBlock::File {
                url,
            } => {
                if raw {
                    serde_json::to_value(url.clone())
                } else {
                    let url_key = config.url_key.as_ref().ok_or_else(|| Error::SerializationFailed {
                        reason: "`url_key` is required for media blocks".to_string(),
                    })?;
                    serde_json::to_value(IndexMap::from([
                        (config.type_key.clone(), type_name),
                        (url_key.clone(), url.clone()),
                    ]))
                }
            },
            OriginalContentBlock::Reasoning {
                value,
            } => serde_json::to_value(value.clone()),
            OriginalContentBlock::ToolCall {
                value,
            } => serde_json::to_value(value.clone()),
            OriginalContentBlock::ToolCallCandidate {
                value,
            } => serde_json::to_value(value.clone()),
            OriginalContentBlock::ToolCallResult {
                identifier: _,
                name: _,
                value,
            } => serde_json::to_value(value.clone()),
            OriginalContentBlock::Translation {
                input,
                source_language_code,
                target_language_code,
            } => {
                let source_key = config
                    .custom_keys
                    .get("source_language_code")
                    .cloned()
                    .unwrap_or_else(|| "source_language_code".to_string());
                let target_key = config
                    .custom_keys
                    .get("target_language_code")
                    .cloned()
                    .unwrap_or_else(|| "target_language_code".to_string());
                let mut map = IndexMap::new();
                match input {
                    TranslationInput::Text {
                        text,
                    } => {
                        map.insert(config.type_key.clone(), Value::String(config.text_key.clone()));
                        map.insert(config.text_key.clone(), Value::String(text.clone()));
                    },
                    TranslationInput::Image {
                        url,
                    } => {
                        let url_key = config.url_key.as_ref().ok_or_else(|| Error::SerializationFailed {
                            reason: "`url_key` is required for image translation".to_string(),
                        })?;
                        map.insert(config.type_key.clone(), Value::String("image".to_string()));
                        map.insert(url_key.clone(), Value::String(url.clone()));
                    },
                }
                map.insert(source_key, Value::String(source_language_code.clone()));
                map.insert(target_key, Value::String(target_language_code.clone()));
                serde_json::to_value(map)
            },
            OriginalContentBlock::Custom {
                value,
            } => serde_json::to_value(value.clone()),
        }
        .map_err(|error| Error::SerializationFailed {
            reason: error.to_string(),
        })?;

        Ok(value)
    }
}
