use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use uzu::types::{
    basic::{ToolCall, ToolDescription, ToolFunction, ToolNamespace, Value},
    session::chat::{ChatContentBlock, ChatMessage, ChatReplyFinishReason, ChatRole},
};

use super::chat_completions::{ChatCompletionRequest, OaiMessage, OaiRole};
use crate::server::chat_tool_calls_errors::{
    ParallelToolCallsError, ToolChoiceError, ToolDefinitionError, ToolHistoryError,
};

pub(super) static TOOL_KIND_FUNCTION: &str = "function";
pub(super) static INCOMPLETE_TOOL_CALL_BATCH_ERROR: &str = "Model generated an incomplete tool-call batch";

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OaiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OaiToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OaiToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
pub struct OaiTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OaiToolFunction,
}

#[derive(Debug, Deserialize)]
pub struct OaiToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ToolChoice {
    Mode(String),
    Function {
        #[serde(rename = "type")]
        kind: String,
        function: ToolChoiceFunction,
    },
}

#[derive(Deserialize)]
struct ToolChoiceFunction {
    name: String,
}

#[derive(Serialize)]
pub(super) struct StreamToolCall {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    function: StreamToolCallFunction,
}

#[derive(Serialize)]
struct StreamToolCallFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

pub(super) fn select_tools(request: &ChatCompletionRequest) -> Result<Vec<&OaiTool>, ToolChoiceError> {
    let tools = request.tools.as_deref().unwrap_or_default();
    let choice = match &request.tool_choice {
        Some(choice) => serde_json::from_value::<ToolChoice>(choice.clone())
            .map_err(|error| ToolChoiceError::Invalid(error.to_string()))?,
        None => ToolChoice::Mode("auto".to_string()),
    };

    match choice {
        ToolChoice::Mode(mode) if mode == "auto" => Ok(tools.iter().collect()),
        ToolChoice::Mode(mode) if mode == "none" => Ok(Vec::new()),
        ToolChoice::Mode(mode) if mode == "required" => Err(ToolChoiceError::Unsupported(mode)),
        ToolChoice::Mode(mode) => Err(ToolChoiceError::Invalid(format!("unknown mode {mode:?}"))),
        ToolChoice::Function {
            kind,
            function,
        } if kind == TOOL_KIND_FUNCTION => {
            if tools.iter().any(|tool| tool.kind == TOOL_KIND_FUNCTION && tool.function.name == function.name) {
                Err(ToolChoiceError::Unsupported(format!("function {}", function.name)))
            } else {
                Err(ToolChoiceError::UnknownFunction(function.name))
            }
        },
        ToolChoice::Function {
            kind,
            ..
        } => Err(ToolChoiceError::Unsupported(kind)),
    }
}

pub(super) fn validate_selected_tools(
    request: &ChatCompletionRequest,
    selected_tools: &[&OaiTool],
) -> Result<(), ToolDefinitionError> {
    let declared_tools = request.tools.as_deref().unwrap_or_default();
    for (index, tool) in declared_tools.iter().enumerate() {
        if tool.kind != TOOL_KIND_FUNCTION {
            return Err(ToolDefinitionError::UnsupportedKind {
                index,
                kind: tool.kind.clone(),
            });
        }
    }
    for tool in selected_tools {
        let index = declared_tools.iter().position(|declared| std::ptr::eq(declared, *tool)).unwrap_or_default();
        if tool.function.strict == Some(true) {
            return Err(ToolDefinitionError::StrictUnsupported {
                index,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_parallel_tool_calls(
    request: &ChatCompletionRequest,
    selected_tools: &[&OaiTool],
    supports_multiple_tool_calls: bool,
) -> Result<(), ParallelToolCallsError> {
    if request.parallel_tool_calls == Some(false) && !selected_tools.is_empty() && supports_multiple_tool_calls {
        return Err(ParallelToolCallsError);
    }
    Ok(())
}

pub(super) fn validate_tool_capability(
    selected_tools: &[&OaiTool],
    supports_tool_calls: bool,
) -> Result<(), ToolDefinitionError> {
    if !selected_tools.is_empty() && !supports_tool_calls {
        return Err(ToolDefinitionError::ToolsUnsupported);
    }
    Ok(())
}

pub(super) fn validate_tool_history(
    messages: &[OaiMessage],
    supports_tool_calls: bool,
    supports_multiple_tool_calls: bool,
) -> Result<(), ToolHistoryError> {
    let mut pending_call_ids = HashSet::<String>::new();
    let mut pending_origin = None;

    for (index, message) in messages.iter().enumerate() {
        if let OaiRole::Unsupported(role) = &message.role {
            return Err(ToolHistoryError {
                message: format!("message role {role:?} is not supported"),
                param: format!("messages[{index}].role"),
                code: "invalid_message_role",
            });
        }
        if !pending_call_ids.is_empty() && message.role != OaiRole::Tool {
            let origin = pending_origin.unwrap_or(index);
            return Err(ToolHistoryError {
                message: "assistant tool calls must all have tool-result messages before the conversation continues"
                    .to_string(),
                param: format!("messages[{origin}].tool_calls"),
                code: "incomplete_tool_history",
            });
        }

        let tool_calls = message.tool_calls.as_deref().unwrap_or_default();
        if !tool_calls.is_empty() && message.role != OaiRole::Assistant {
            return Err(ToolHistoryError {
                message: "tool_calls are only valid on assistant messages".to_string(),
                param: format!("messages[{index}].tool_calls"),
                code: "invalid_tool_history",
            });
        }
        if !tool_calls.is_empty() && !supports_tool_calls {
            return Err(ToolHistoryError {
                message: "tool-call history is not supported by the loaded model".to_string(),
                param: format!("messages[{index}].tool_calls"),
                code: "unsupported_tool_history",
            });
        }
        if tool_calls.len() > 1 && !supports_multiple_tool_calls {
            return Err(ToolHistoryError {
                message: "multiple tool calls in one history message are not supported by the loaded model".to_string(),
                param: format!("messages[{index}].tool_calls"),
                code: "unsupported_parallel_tool_history",
            });
        }
        let requires_content = matches!(&message.role, OaiRole::System | OaiRole::Developer | OaiRole::User)
            || (message.role == OaiRole::Assistant && tool_calls.is_empty());
        if requires_content && message.content.is_none() {
            return Err(ToolHistoryError {
                message: format!("{} messages require content", message.role),
                param: format!("messages[{index}].content"),
                code: "invalid_message_content",
            });
        }

        for (call_index, tool_call) in tool_calls.iter().enumerate() {
            let prefix = format!("messages[{index}].tool_calls[{call_index}]");
            if tool_call.kind != TOOL_KIND_FUNCTION {
                return Err(ToolHistoryError {
                    message: format!("historical tool-call type {:?} is not supported", tool_call.kind),
                    param: format!("{prefix}.type"),
                    code: "unsupported_tool_type",
                });
            }
            if tool_call.id.is_empty() {
                return Err(ToolHistoryError {
                    message: "historical tool-call id must not be empty".to_string(),
                    param: format!("{prefix}.id"),
                    code: "invalid_tool_history",
                });
            }
            if let Err(error) = serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments) {
                return Err(ToolHistoryError {
                    message: format!("historical tool-call arguments are not valid JSON: {error}"),
                    param: format!("{prefix}.function.arguments"),
                    code: "invalid_tool_history",
                });
            }
            if !pending_call_ids.insert(tool_call.id.clone()) {
                return Err(ToolHistoryError {
                    message: format!("duplicate historical tool-call id {:?}", tool_call.id),
                    param: format!("{prefix}.id"),
                    code: "invalid_tool_history",
                });
            }
        }
        if !tool_calls.is_empty() {
            pending_origin = Some(index);
        }

        if message.role == OaiRole::Tool {
            if !supports_tool_calls {
                return Err(ToolHistoryError {
                    message: "tool-result history is not supported by the loaded model".to_string(),
                    param: format!("messages[{index}].tool_call_id"),
                    code: "unsupported_tool_history",
                });
            }
            let Some(identifier) = message.tool_call_id.as_deref().filter(|identifier| !identifier.is_empty()) else {
                return Err(ToolHistoryError {
                    message: "tool messages require a nonempty tool_call_id".to_string(),
                    param: format!("messages[{index}].tool_call_id"),
                    code: "invalid_tool_history",
                });
            };
            if message.content.is_none() {
                return Err(ToolHistoryError {
                    message: "tool messages require content".to_string(),
                    param: format!("messages[{index}].content"),
                    code: "invalid_tool_history",
                });
            }
            if !pending_call_ids.remove(identifier) {
                return Err(ToolHistoryError {
                    message: format!("tool_call_id {identifier:?} does not match an outstanding assistant tool call"),
                    param: format!("messages[{index}].tool_call_id"),
                    code: "invalid_tool_history",
                });
            }
            if pending_call_ids.is_empty() {
                pending_origin = None;
            }
        }
    }

    if !pending_call_ids.is_empty() {
        let origin = pending_origin.unwrap_or_default();
        return Err(ToolHistoryError {
            message: "assistant tool calls are missing tool-result messages".to_string(),
            param: format!("messages[{origin}].tool_calls"),
            code: "incomplete_tool_history",
        });
    }
    Ok(())
}

pub(super) fn to_chat_messages<'a>(
    messages: &[OaiMessage],
    tools: impl IntoIterator<Item = &'a OaiTool>,
) -> Vec<ChatMessage> {
    let mut tool_names = std::collections::HashMap::<String, String>::new();
    let mut chat_messages = messages
        .iter()
        .map(|message| {
            let role = match &message.role {
                OaiRole::System => ChatRole::System {},
                OaiRole::Developer => ChatRole::Developer {},
                OaiRole::User => ChatRole::User {},
                OaiRole::Assistant => ChatRole::Assistant {},
                OaiRole::Tool => ChatRole::Tool {},
                OaiRole::Unsupported(_) => unreachable!("message roles must be validated before conversion"),
            };
            let mut chat_message = ChatMessage::for_role(role.clone());

            if role == (ChatRole::Tool {}) {
                let identifier = message.tool_call_id.clone();
                let name = identifier.as_ref().and_then(|identifier| tool_names.get(identifier).cloned());
                let value = message
                    .content
                    .as_deref()
                    .and_then(|content| serde_json::from_str(content).ok())
                    .unwrap_or_else(|| serde_json::Value::String(message.content.clone().unwrap_or_default()));
                chat_message = chat_message.with_block(ChatContentBlock::ToolCallResult {
                    identifier,
                    name,
                    value: Value::from(value),
                });
            } else {
                if let Some(content) = &message.content {
                    chat_message = chat_message.with_text(content.clone());
                }
                if role == (ChatRole::Assistant {}) {
                    for tool_call in message.tool_calls.as_deref().unwrap_or_default() {
                        tool_names.insert(tool_call.id.clone(), tool_call.function.name.clone());
                        chat_message = chat_message.with_tool_call(ToolCall {
                            identifier: Some(tool_call.id.clone()),
                            name: tool_call.function.name.clone(),
                            arguments: Value {
                                json: tool_call.function.arguments.clone(),
                            },
                        });
                    }
                }
            }
            chat_message
        })
        .collect::<Vec<_>>();

    let descriptions = tools
        .into_iter()
        .map(|tool| ToolDescription::Function {
            tool_function: ToolFunction {
                name: tool.function.name.clone(),
                description: tool.function.description.clone().unwrap_or_default(),
                parameters: tool.function.parameters.clone().map(Value::from),
                return_definition: None,
            },
        })
        .collect::<Vec<_>>();
    if !descriptions.is_empty() {
        let namespaces = vec![ToolNamespace {
            name: "functions".to_string(),
            description: None,
            tools: descriptions,
        }];
        if let Some(developer_message) =
            chat_messages.iter_mut().find(|message| message.role == (ChatRole::Developer {}))
        {
            developer_message.content.push(ChatContentBlock::Tools {
                namespaces,
            });
        } else {
            let definitions = ChatMessage::developer().with_tool_namespaces(namespaces);
            let position = chat_messages.iter().position(|message| message.role == (ChatRole::System {}));
            chat_messages.insert(position.map(|position| position + 1).unwrap_or(0), definitions);
        }
    }

    chat_messages
}

pub(super) fn response_tool_calls(message: &ChatMessage) -> Option<Vec<OaiToolCall>> {
    let tool_calls = message
        .tool_calls()
        .into_iter()
        .map(|tool_call| OaiToolCall {
            id: tool_call.identifier.unwrap_or_default(),
            kind: TOOL_KIND_FUNCTION.to_string(),
            function: OaiToolCallFunction {
                name: tool_call.name,
                arguments: tool_call.arguments.json,
            },
        })
        .collect::<Vec<_>>();
    (!tool_calls.is_empty()).then_some(tool_calls)
}

pub(super) fn has_incomplete_tool_call_batch(
    message: &ChatMessage,
    finish_reason: Option<&ChatReplyFinishReason>,
) -> bool {
    let has_candidate = message.content.iter().any(|block| matches!(block, ChatContentBlock::ToolCallCandidate { .. }));
    if has_candidate {
        return true;
    }
    let has_calls = !message.tool_calls().is_empty();
    match finish_reason {
        Some(ChatReplyFinishReason::ToolCalls) => !has_calls,
        Some(_) => has_calls,
        None => false,
    }
}

pub(super) fn tool_call_batch_error(
    message: &ChatMessage,
    finish_reason: Option<&ChatReplyFinishReason>,
    allowed_tool_names: &[String],
    is_terminal: bool,
) -> Option<String> {
    if has_incomplete_tool_call_batch(message, finish_reason)
        || (is_terminal && finish_reason.is_none() && !message.tool_calls().is_empty())
    {
        return Some(INCOMPLETE_TOOL_CALL_BATCH_ERROR.to_string());
    }

    let mut identifiers = HashSet::<String>::new();
    for tool_call in message.tool_calls() {
        if !allowed_tool_names.contains(&tool_call.name) {
            return Some(format!("Model generated a call to undeclared tool {:?}", tool_call.name));
        }
        let Some(identifier) = tool_call.identifier.as_deref().filter(|identifier| !identifier.is_empty()) else {
            return Some("Model generated a tool call without an identifier".to_string());
        };
        if !identifiers.insert(identifier.to_string()) {
            return Some(format!("Model generated duplicate tool-call identifier {identifier:?}"));
        }
        if let Err(error) = serde_json::from_str::<serde_json::Value>(&tool_call.arguments.json) {
            return Some(format!("Model generated invalid JSON tool-call arguments: {error}"));
        }
    }
    None
}

pub(super) fn normalize_tool_calls_for_capability(
    message: &ChatMessage,
    supports_multiple_tool_calls: bool,
) -> ChatMessage {
    let mut message = message.clone();
    if !supports_multiple_tool_calls && message.tool_calls().len() > 1 {
        let mut tool_call_seen = false;
        message.content.retain(|block| match block {
            ChatContentBlock::ToolCall {
                ..
            }
            | ChatContentBlock::ToolCallCandidate {
                ..
            } => !std::mem::replace(&mut tool_call_seen, true),
            _ => true,
        });
    }
    message
}

pub(super) fn stream_tool_call_deltas(
    tool_calls: &[ToolCall],
    emitted_count: &mut usize,
) -> Vec<StreamToolCall> {
    let deltas = tool_calls
        .iter()
        .enumerate()
        .skip(*emitted_count)
        .map(|(index, tool_call)| StreamToolCall {
            index,
            id: Some(tool_call.identifier.clone().unwrap_or_default()),
            kind: Some(TOOL_KIND_FUNCTION.to_string()),
            function: StreamToolCallFunction {
                name: Some(tool_call.name.clone()),
                arguments: Some(tool_call.arguments.json.clone()),
            },
        })
        .collect::<Vec<_>>();
    *emitted_count = tool_calls.len();
    deltas
}

#[cfg(test)]
#[path = "../../unit/server/chat_tools_calls_test.rs"]
mod tests;
