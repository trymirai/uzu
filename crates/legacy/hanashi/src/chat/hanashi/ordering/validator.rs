use shoji::types::session::chat::{ChatContentBlock, ChatMessage, ChatRole};

use crate::chat::hanashi::ordering::{Config, Error};

enum ToolCallValidationMode {
    History,
    Streamed,
}

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
        let count = Self::validate_tool_call_batches(messages, ToolCallValidationMode::History)?;
        if count > 0 {
            return Err(Error::UnresolvedToolCalls {
                count,
            });
        }
        Ok(())
    }

    /// Validate assistant transitions and return the number of unresolved calls and candidates.
    /// A generated reply may end with calls or candidates for the caller to handle.
    pub fn validate_streamed_tool_calls<'a>(
        messages: impl IntoIterator<Item = &'a ChatMessage>
    ) -> Result<usize, Error> {
        Self::validate_tool_call_batches(messages, ToolCallValidationMode::Streamed)
    }

    fn validate_tool_call_batches<'a>(
        messages: impl IntoIterator<Item = &'a ChatMessage>,
        mode: ToolCallValidationMode,
    ) -> Result<usize, Error> {
        let mut pending: Vec<(Option<&str>, Option<&str>)> = Vec::new();
        let mut pending_candidates = 0;
        for message in messages {
            match message.role {
                ChatRole::Assistant {} => {
                    let count = pending.len() + pending_candidates;
                    if count > 0 {
                        return Err(Error::UnresolvedToolCalls {
                            count,
                        });
                    }
                    // Execution abandons the entire batch if any call could not be finalized.
                    // Historical batches like this must not block a later external retry.
                    if matches!(mode, ToolCallValidationMode::History)
                        && message
                            .content
                            .iter()
                            .any(|block| matches!(block, ChatContentBlock::ToolCallCandidate { .. }))
                    {
                        continue;
                    }
                    for block in &message.content {
                        match block {
                            ChatContentBlock::ToolCall {
                                value,
                            } => pending.push((value.identifier.as_deref(), Some(value.name.as_str()))),
                            // Within one generation, candidates must block a subsequent assistant
                            // frame, even if tool results resolve the finished calls in the batch.
                            ChatContentBlock::ToolCallCandidate {
                                ..
                            } => pending_candidates += 1,
                            _ => {},
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
        Ok(pending.len() + pending_candidates)
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
