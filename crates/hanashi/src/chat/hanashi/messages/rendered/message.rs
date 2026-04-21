use std::collections::HashMap;

use indexmap::IndexMap;
use serde_json::{Map, Value};
use shoji::types::session::chat::{ContentBlockType, Message as OriginalMessage, Role};

use crate::chat::hanashi::messages::{
    Error,
    canonical::{Config as CanonicalConfig, Message as CanonicalMessage},
    rendered::{Config, Field, FieldConfig},
};

pub struct Message {
    pub message: Map<String, Value>,
    pub context: Map<String, Value>,
    pub metadata: HashMap<String, Value>,
}

impl Message {
    pub fn from_message(
        original_message: &OriginalMessage,
        canonical_config: &CanonicalConfig,
        rendered_configs: &IndexMap<Role, Config>,
    ) -> Result<Self, Error> {
        let role = &original_message.role;

        let config = rendered_configs.get(role).ok_or_else(|| Error::UnsupportedRole {
            role: role.clone(),
        })?;

        // Validate no duplicate block types across fields
        let mut seen_block_types: IndexMap<ContentBlockType, String> = IndexMap::new();
        for (field_name, field) in config.message.iter().chain(config.context.iter()) {
            let block_types = match &field.config {
                FieldConfig::Unique {
                    block,
                    ..
                } => vec![block.clone()],
                FieldConfig::Collected {
                    blocks,
                    ..
                } => blocks.clone(),
            };
            for block_type in block_types {
                if let Some(existing_field) = seen_block_types.get(&block_type) {
                    if existing_field != field_name {
                        return Err(Error::DuplicateBlock {
                            role: role.clone(),
                            block_type,
                        });
                    }
                }
                seen_block_types.insert(block_type, field_name.clone());
            }
        }

        // Determine which block types should get raw canonical serialization
        let mut raw_types: Vec<ContentBlockType> = Vec::new();
        for (_, field) in config.message.iter().chain(config.context.iter()) {
            if field.disable_raw {
                continue;
            }
            match &field.config {
                FieldConfig::Unique {
                    block,
                    ..
                } => {
                    raw_types.push(block.clone());
                },
                FieldConfig::Collected {
                    blocks,
                    ..
                } => {
                    let matching_count =
                        original_message.content.iter().filter(|block| blocks.contains(&block.get_type())).count();
                    if matching_count == 1 {
                        for block in &original_message.content {
                            if blocks.contains(&block.get_type()) {
                                raw_types.push(block.get_type());
                            }
                        }
                    }
                },
            }
        }

        // Canonicalize
        let canonical_message = CanonicalMessage::from_message(original_message, canonical_config, &raw_types)?;

        let mut message: Map<String, Value> = Map::new();
        let mut context: Map<String, Value> = Map::new();
        let mut seen_fields: IndexMap<String, usize> = IndexMap::new();

        let destinations: &mut [(&IndexMap<String, Field>, &mut Map<String, Value>)] =
            &mut [(&config.message, &mut message), (&config.context, &mut context)];

        for canonical_block in &canonical_message.content {
            let block_type = &canonical_block.r#type;
            let value = &canonical_block.value;

            let mut matched = false;
            for (fields, destination) in destinations.iter_mut() {
                if let Some((field_name, field)) = find_field_for_block(fields, block_type) {
                    *seen_fields.entry(field_name.clone()).or_insert(0) += 1;
                    let raw = raw_types.contains(block_type);
                    apply_field_value(destination, &field_name, &field.config, value, raw, role, block_type)?;
                    matched = true;
                    break;
                }
            }

            if !matched {
                return Err(Error::UnsupportedBlock {
                    role: role.clone(),
                    block_type: block_type.clone(),
                });
            }
        }

        // Check required fields
        for (field_name, field) in config.message.iter().chain(config.context.iter()) {
            if field.required && !seen_fields.contains_key(field_name) {
                return Err(Error::FieldRequired {
                    role: role.clone(),
                    field: field_name.clone(),
                });
            }
        }

        // Insert role
        if !message.is_empty() {
            message.insert(canonical_config.role_key.clone(), Value::String(config.role.clone()));
        }

        Ok(Self {
            message,
            context,
            metadata: canonical_message.metadata.clone(),
        })
    }
}

fn find_field_for_block<'a>(
    fields: &'a IndexMap<String, Field>,
    block_type: &ContentBlockType,
) -> Option<(String, &'a Field)> {
    fields.iter().find_map(|(name, field)| {
        let matches = match &field.config {
            FieldConfig::Unique {
                block,
                ..
            } => block == block_type,
            FieldConfig::Collected {
                blocks,
                ..
            } => blocks.contains(block_type),
        };
        if matches {
            Some((name.clone(), field))
        } else {
            None
        }
    })
}

fn apply_field_value(
    destination: &mut Map<String, Value>,
    field_name: &str,
    config: &FieldConfig,
    value: &Value,
    raw: bool,
    role: &Role,
    block_type: &ContentBlockType,
) -> Result<(), Error> {
    match config {
        FieldConfig::Unique {
            allowed_values,
            mapping,
            ..
        } => {
            if destination.contains_key(field_name) {
                return Err(Error::MultipleNotAllowed {
                    role: role.clone(),
                    block_type: block_type.clone(),
                });
            }

            if let Some(allowed) = allowed_values {
                if !allowed.contains(value) {
                    return Err(Error::ValueNotAllowed {
                        role: role.clone(),
                        block_type: block_type.clone(),
                        allowed_values: format!("{:?}", allowed),
                    });
                }
            }

            if let Some(mapping) = mapping {
                let lookup_key = value.as_str().ok_or_else(|| Error::SerializationFailed {
                    reason: format!("Expected string for mapping lookup, got {value}"),
                })?;
                let mapped_value = mapping.get(lookup_key).ok_or_else(|| Error::UnmappedValue {
                    role: role.clone(),
                    block_type: block_type.clone(),
                    value: lookup_key.to_string(),
                })?;
                if let Some(mapped_value) = mapped_value {
                    destination.insert(field_name.to_string(), mapped_value.clone());
                }
            } else {
                destination.insert(field_name.to_string(), value.clone());
            }
        },
        FieldConfig::Collected {
            limit,
            ..
        } => {
            if raw {
                destination.insert(field_name.to_string(), value.clone());
            } else {
                let entry = destination.entry(field_name.to_string()).or_insert_with(|| Value::Array(vec![]));
                if let Value::Array(array) = entry {
                    if let Some(limit) = limit {
                        if array.len() >= *limit {
                            return Err(Error::LimitExceeded {
                                role: role.clone(),
                                block_type: block_type.clone(),
                                limit: *limit,
                            });
                        }
                    }
                    array.push(value.clone());
                }
            }
        },
    }
    Ok(())
}
