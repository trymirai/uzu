use shoji::types::session::chat::{ChatContentBlock, ChatMessage, ChatRole};

use crate::chat::hanashi::ordering::{Config, Error};

pub struct Validator {
    current: Option<ChatRole>,
    config: Config,
}

impl Validator {
    pub fn new(config: Config) -> Self {
        Self {
            current: None,
            config,
        }
    }

    pub fn reset(&mut self) {
        self.current = None;
    }

    /// Check complete input messages, including calls whose IDs were assigned after decoding.
    /// An empty pending list at the end allows the next generated assistant reply.
    pub fn validate_tool_calls<'a>(messages: impl IntoIterator<Item = &'a ChatMessage>) -> Result<(), Error> {
        let count = Self::validate_streamed_tool_calls(messages)?;
        if count > 0 {
            return Err(Error::UnresolvedToolCalls {
                count,
            });
        }
        Ok(())
    }

    /// Validate assistant transitions and return the number of calls still awaiting results.
    /// A generated reply may end with pending calls for the caller to execute.
    pub fn validate_streamed_tool_calls<'a>(
        messages: impl IntoIterator<Item = &'a ChatMessage>
    ) -> Result<usize, Error> {
        let mut pending: Vec<(Option<&str>, Option<&str>)> = Vec::new();
        for message in messages {
            match message.role {
                ChatRole::Assistant {} => {
                    if !pending.is_empty() {
                        return Err(Error::UnresolvedToolCalls {
                            count: pending.len(),
                        });
                    }
                    // Candidates cannot be executed and must not block a retry.
                    for block in &message.content {
                        if let ChatContentBlock::ToolCall {
                            value,
                        } = block
                        {
                            pending.push((value.identifier.as_deref(), Some(value.name.as_str())));
                        }
                    }
                },
                ChatRole::Tool {} => {
                    for block in &message.content {
                        let ChatContentBlock::ToolCallResult {
                            identifier,
                            name,
                            ..
                        } = block
                        else {
                            continue;
                        };
                        // Match IDs first. Formats without IDs can match by name, or by order
                        // when neither the call nor its result provides a name.
                        let position = identifier
                            .as_deref()
                            .and_then(|id| pending.iter().position(|(pending_id, _)| *pending_id == Some(id)))
                            .or_else(|| {
                                pending.iter().position(|(pending_id, pending_name)| {
                                    (identifier.is_none() || pending_id.is_none())
                                        && name.as_deref().is_none_or(|name| {
                                            pending_name.is_none_or(|pending_name| name == pending_name)
                                        })
                                })
                            });
                        if let Some(position) = position {
                            pending.remove(position);
                        }
                    }
                },
                _ => {},
            }
        }
        Ok(pending.len())
    }

    pub fn validate_next(
        &mut self,
        role: &ChatRole,
    ) -> Result<(), Error> {
        if matches!(role, ChatRole::Custom { .. }) {
            return Ok(());
        }

        match &self.current {
            None => {
                if !self.config.initial.contains(role) {
                    return Err(Error::InvalidInitial {
                        expected: format_roles(&self.config.initial),
                        got: role.clone(),
                    });
                }
            },
            Some(current) => {
                let allowed = self.config.transitions.get(current).ok_or_else(|| Error::NoTransitions {
                    role: current.clone(),
                })?;
                if !allowed.contains(role) {
                    return Err(Error::InvalidTransition {
                        after: current.to_string(),
                        expected: format_roles(allowed),
                        got: role.clone(),
                    });
                }
            },
        }
        self.current = Some(role.clone());
        Ok(())
    }
}

fn format_roles(roles: &[ChatRole]) -> String {
    roles.iter().map(|role| role.to_string()).collect::<Vec<_>>().join(", ")
}
